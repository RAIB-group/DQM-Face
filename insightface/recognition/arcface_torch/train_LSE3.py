"""
Final SOTA Booster: LSE-Residual Attention
Includes: MLP Baseline + LSE Facet Booster + Norm-Awareness + Slurm Auto-Init
"""

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
# ADVANCED ATTENTION MODULE
# ============================================================

class LSEResidualAttention(nn.Module):
    def __init__(self, in_features, reduction=128, num_facets=4):
        super(LSEResidualAttention, self).__init__()
        self.num_facets = num_facets
        self.mlp_path = nn.Sequential(
            nn.Linear(in_features, reduction),
            nn.PReLU(),
            nn.Linear(reduction, 1)
        )
        self.facet_projection = nn.Linear(in_features + 1, reduction * num_facets)
        self.facet_relu = nn.PReLU()
        self.facet_score = nn.Linear(reduction, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        s_mlp = self.mlp_path(x) 
        x_with_norm = torch.cat([x, norm], dim=1)
        f = self.facet_projection(x_with_norm)
        f = self.facet_relu(f)
        f = f.view(x.shape[0], self.num_facets, -1) 
        f_scores = self.facet_score(f) 
        s_lse = torch.logsumexp(f_scores, dim=1) 
        return 0.5 + self.sigmoid(s_mlp + s_lse)

# ============================================================
# LOSS MODULE
# ============================================================

class AttenDualPartialFC(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=64.0, m1=0.45, m2=0.05):
        super(AttenDualPartialFC, self).__init__()
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.attention_net = LSEResidualAttention(embedding_size)
        self.scale, self.m1, self.m2 = scale, m1, m2
        self.last_attn = 0.0

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        attn = self.attention_net(embeddings)
        self.last_attn = attn.mean().item()
        m1_d = self.m1 * attn
        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
        output = (one_hot * torch.cos(theta + m1_d)) + ((1.0 - one_hot) * torch.cos(theta - self.m2))
        return F.cross_entropy(output * self.scale, labels)

# ============================================================
# VALIDATION UTILS
# ============================================================

@torch.no_grad()
def validate(backbone, data_path, val_targets):
    backbone.eval()
    results = {}
    for name in val_targets:
        bin_path = os.path.join(data_path, f"{name}.bin")
        if not os.path.exists(bin_path): continue
        with open(bin_path, 'rb') as f: bins, issame = pickle.load(f, encoding='bytes')
        embs = []
        for flip in [0, 1]:
            data = []
            for i in range(len(issame)*2):
                import cv2
                img = cv2.imdecode(np.frombuffer(bins[i], np.uint8), cv2.IMREAD_COLOR)
                if flip: img = cv2.flip(img, 1)
                img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float()
                data.append((img / 255.0 - 0.5) / 0.5)
            data = torch.stack(data)
            batches = [backbone(data[j:j+64].cuda()).cpu().numpy() for j in range(0, len(data), 64)]
            embs.append(np.concatenate(batches))
        embeddings = normalize(embs[0] + embs[1])
        emb1, emb2 = embeddings[0::2], embeddings[1::2]
        dist = np.sum(np.square(emb1 - emb2), 1)
        acc = max([np.mean(np.less(dist, t) == np.array(issame)) for t in np.arange(0, 4, 0.01)])
        results[name] = acc
        logging.info(f"  [Val] {name}: {acc*100:.2f}%")
    backbone.train(); return results

# ============================================================
# TRAINING Logic
# ============================================================

def train(cfg):
    # Setup Slurm Environment
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29505")

    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    if rank == 0: os.makedirs(cfg.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank==0 else logging.ERROR)
    
    writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tb")) if rank == 0 and cfg.tensorboard else None

    from backbones import get_model
    from dataset import get_dataloader

    train_loader = get_dataloader(cfg.rec, local_rank, cfg.batch_size, False, False, cfg.seed, 8)
    backbone = get_model(cfg.network, dropout=0.0, fp16=True, num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank])
    
    module_fc = AttenDualPartialFC(cfg.embedding_size, 85742, 64.0, cfg.m1, cfg.m2).cuda()
    opt = torch.optim.SGD([{'params': backbone.parameters()}, {'params': module_fc.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    
    steps_per_epoch = cfg.num_image // (cfg.batch_size * world_size)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m * steps_per_epoch for m in cfg.milestones], gamma=0.1)
    amp = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.num_epoch):
        module_fc.m2 = 0.02 if epoch < 5 else 0.05
        train_loader.sampler.set_epoch(epoch)
        for i, (img, labels) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                loss = module_fc(backbone(img.cuda()), labels.cuda())
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            amp.step(opt); amp.update(); opt.zero_grad(); lr_scheduler.step()
            
            if i % 200 == 0 and rank == 0:
                logging.info(f"E[{epoch}] S[{i}] Loss: {loss.item():.4f} Attn: {module_fc.last_attn:.3f}")
                if writer: writer.add_scalar("train/loss", loss.item(), epoch * steps_per_epoch + i)

        validate(backbone.module, cfg.rec, cfg.val_targets)
        if rank == 0: torch.save(backbone.module.state_dict(), os.path.join(cfg.output, f"model_e{epoch}.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="lse_attention")
    parser.add_argument("--tensorboard", action="store_true") # Fixed Argparse Error
    args = parser.parse_args()

    # NumPy fix
    for a in ['bool', 'int', 'float']: 
        if not hasattr(np, a): setattr(np, a, getattr(np, f"{a}_"))

    cfg = edict({
        'network': "r100", 'embedding_size': 512, 'batch_size': 256, 'lr': 0.1,
        'num_epoch': 25, 'milestones': [10, 18, 22], 'm1': 0.45, 'm2': 0.05,
        'rec': "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore",
        'val_targets': ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"],
        'output': f"./output/LSE_Run_{datetime.now().strftime('%H%M%S')}",
        'num_image': 5822653, 'seed': 2048, 'tensorboard': args.tensorboard
    })
    train(cfg)

if __name__ == "__main__":
    main()