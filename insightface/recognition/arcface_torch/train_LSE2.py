import argparse
import logging
import os
import json
import pickle
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import distributed
from torch.utils.data import DataLoader
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
import sklearn
from sklearn.preprocessing import normalize

# ============================================================
# ADVANCED ATTENTION MODULE (LSE-RESIDUAL + NORM-AWARE)
# ============================================================

class LSEResidualAttention(nn.Module):
    """
    SOTA ENHANCEMENT:
    1. Residual MLP: Keeps your successful Method 1011 logic.
    2. LSE Facets: Detects specific difficulty (Age/Pose) using Log-Sum-Exp.
    3. Norm-Aware: Uses embedding norm as a quality proxy (AdaFace/MagFace trick).
    """
    def __init__(self, in_features, reduction=128, num_facets=4):
        super(LSEResidualAttention, self).__init__()
        self.num_facets = num_facets
        
        # Branch 1: Your original successful MLP logic
        self.mlp_path = nn.Sequential(
            nn.Linear(in_features, reduction),
            nn.PReLU(),
            nn.Linear(reduction, 1)
        )
        
        # Branch 2: LSE Facet logic (Norm-Aware)
        # We add +1 to in_features for the Feature Norm
        self.facet_projection = nn.Linear(in_features + 1, reduction * num_facets)
        self.facet_relu = nn.PReLU()
        self.facet_score = nn.Linear(reduction, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Calculate Feature Norm
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        
        # Path 1: Baseline MLP
        s_mlp = self.mlp_path(x) 
        
        # Path 2: LSE Facets
        x_with_norm = torch.cat([x, norm], dim=1)
        f = self.facet_projection(x_with_norm)
        f = self.facet_relu(f)
        f = f.view(x.shape[0], self.num_facets, -1) 
        
        f_scores = self.facet_score(f) 
        # Smooth Max over facets
        s_lse = torch.logsumexp(f_scores, dim=1) 
        
        # Combine Paths
        total_score = s_mlp + s_lse
        
        # Scale to [0.5, 1.5]
        return 0.5 + self.sigmoid(total_score)

# ============================================================
# LOSS MODULE
# ============================================================

class AttenDualPartialFC(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=64.0, m1=0.45, m2=0.05):
        super(AttenDualPartialFC, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.m1 = m1
        self.m2 = m2 
        
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.attention_net = LSEResidualAttention(embedding_size)
        self.last_attention_score = None

    def update_m2(self, new_m2):
        self.m2 = new_m2
        
    def forward(self, embeddings, labels):
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings_norm, weight_norm)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # Get the Booster Attention Score
        atten_score = self.attention_net(embeddings)
        self.last_attention_score = atten_score.mean().item()

        # Dynamic Margin Logic
        m1_dynamic = self.m1 * atten_score
        
        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)

        target_angle = theta + m1_dynamic
        target_cosine = torch.cos(target_angle)

        non_target_angle = theta - self.m2
        non_target_cosine = torch.cos(non_target_angle)

        output = (one_hot * target_cosine) + ((1.0 - one_hot) * non_target_cosine)
        output *= self.scale
        return F.cross_entropy(output, labels)

    def get_attention_score(self):
        return self.last_attention_score if self.last_attention_score else 0.0

# ============================================================
# VALIDATION & UTILS
# ============================================================

def evaluate_accuracy(embeddings, issame_list):
    emb1, emb2 = embeddings[0::2], embeddings[1::2]
    dist = np.sum(np.square(emb1 - emb2), 1)
    thresholds = np.arange(0, 4, 0.01)
    accs = [np.mean(np.less(dist, t) == np.array(issame_list)) for t in thresholds]
    return max(accs)

@torch.no_grad()
def validate(backbone, data_path, val_targets):
    backbone.eval()
    results = {}
    for name in val_targets:
        bin_path = os.path.join(data_path, f"{name}.bin")
        if not os.path.exists(bin_path): continue
        with open(bin_path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')
        
        embeddings_list = []
        for flip in [0, 1]:
            data = []
            for i in range(len(issame_list)*2):
                import cv2
                img = cv2.imdecode(np.frombuffer(bins[i], np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if flip: img = cv2.flip(img, 1)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                img = (img / 255.0 - 0.5) / 0.5
                data.append(img)
            data = torch.stack(data)
            batches = [backbone(data[j:j+64].cuda()).cpu().numpy() for j in range(0, len(data), 64)]
            embeddings_list.append(np.concatenate(batches))
        
        embeddings = normalize(embeddings_list[0] + embeddings_list[1])
        acc = evaluate_accuracy(embeddings, issame_list)
        results[name] = acc
        logging.info(f"  [Val] {name}: {acc*100:.2f}%")
    backbone.train()
    return results

# ============================================================
# TRAINING SETUP
# ============================================================

def get_config():
    cfg = edict()
    cfg.network = "r100"
    cfg.embedding_size = 512
    cfg.batch_size = 256
    cfg.lr = 0.1
    cfg.num_epoch = 25
    cfg.milestones = [10, 18, 22]
    cfg.m1, cfg.m2 = 0.45, 0.05 # Refined values for LSE-Booster
    cfg.rec = "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore"
    cfg.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    cfg.output = f"./output/LSE_Booster_R100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg.num_image = 5822653
    cfg.seed = 2048
    return cfg

def train():
    cfg = get_config()
    os.makedirs(cfg.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(cfg.output, "train.log")), logging.StreamHandler()])
    
    # Distributed Setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl")
    
    from backbones import get_model
    from dataset import get_dataloader

    # Init
    train_loader = get_dataloader(cfg.rec, local_rank, cfg.batch_size, False, False, cfg.seed, 8)
    backbone = get_model(cfg.network, dropout=0.0, fp16=True, num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank])
    
    module_fc = AttenDualPartialFC(cfg.embedding_size, 85742, 64.0, cfg.m1, cfg.m2).cuda()
    
    opt = torch.optim.SGD([{'params': backbone.parameters()}, {'params': module_fc.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    
    steps_per_epoch = cfg.num_image // (cfg.batch_size * int(os.environ.get("WORLD_SIZE", 1)))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m * steps_per_epoch for m in cfg.milestones], gamma=0.1)
    amp = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.num_epoch):
        # M2 Schedule (Start low, then increase)
        current_m2 = 0.02 if epoch < 5 else 0.05
        module_fc.update_m2(current_m2)
        
        train_loader.sampler.set_epoch(epoch)
        for img, labels in train_loader:
            img, labels = img.cuda(), labels.cuda()
            with torch.cuda.amp.autocast():
                loss = module_fc(backbone(img), labels)

            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            amp.step(opt)
            amp.update()
            opt.zero_grad()
            lr_scheduler.step()

        # Epoch End
        logging.info(f"Epoch {epoch} complete. m2: {module_fc.m2}")
        validate(backbone.module, cfg.rec, cfg.val_targets)
        if local_rank == 0:
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, "model_latest.pt"))

if __name__ == "__main__":
    train()