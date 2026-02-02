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
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
import sklearn.preprocessing

# ============================================================
# ATTENTION MODULE
# ============================================================

class DualAttention(nn.Module):
    def __init__(self, in_features, **kwargs):
        super(DualAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, 128); self.relu = nn.PReLU()
        self.fc2 = nn.Linear(128, 1); self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return 0.5 + self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# ============================================================
# LOSS MODULE
# ============================================================

class AttenDualPartialFC(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=64.0, m1=0.4, m2=0.02):
        super(AttenDualPartialFC, self).__init__()
        self.embedding_size, self.num_classes, self.scale, self.m1, self.m2 = embedding_size, num_classes, scale, m1, m2
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.attention_net = DualAttention(embedding_size)
        self.last_attn = 0.0

    def update_m2(self, new_m2):
        self.m2 = new_m2
        
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
# VALIDATION Logic
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
        embeddings = sklearn.preprocessing.normalize(embs[0] + embs[1])
        dist = np.sum(np.square(embeddings[0::2] - embeddings[1::2]), 1)
        acc = max([np.mean(np.less(dist, t) == np.array(issame)) for t in np.arange(0, 4, 0.01)])
        results[name] = acc
        logging.info(f"  [Val] {name}: {acc*100:.2f}%")
    backbone.train(); return results

# ============================================================
# TRAINING Logic
# ============================================================

def setup_distributed():
    # --- NCCL CRASH FIXES ---
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    # ------------------------
    
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    
    os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"] = str(rank), str(world_size), str(local_rank)
    if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29515"
    
    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world_size

def train(cfg, rank, local_rank, world_size):
    os.makedirs(cfg.output, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank==0 else logging.ERROR, 
                        handlers=[logging.FileHandler(os.path.join(cfg.output, "train.log")), logging.StreamHandler()])
    
    from backbones import get_model
    from dataset import get_dataloader

    train_loader = get_dataloader(cfg.rec, local_rank, cfg.batch_size, False, False, cfg.seed, 8)
    backbone = get_model(cfg.network, dropout=0.0, fp16=True, num_features=cfg.embedding_size).cuda()
    
    # Only use DDP if we actually have multiple processes to avoid communication errors
    if world_size > 1:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank])
    
    module_fc = AttenDualPartialFC(cfg.embedding_size, 85742, 64.0, cfg.m1, cfg.m2).cuda()
    opt = torch.optim.SGD([{'params': backbone.parameters()}, {'params': module_fc.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    
    steps_per_epoch = cfg.num_image // (cfg.batch_size * world_size)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m * steps_per_epoch for m in cfg.milestones], gamma=0.1)
    amp = torch.cuda.amp.GradScaler()

    for epoch in range(cfg.num_epoch):
        module_fc.update_m2(0.02 if epoch < 10 else 0.05)
        train_loader.sampler.set_epoch(epoch)
        for i, (img, labels) in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                loss = module_fc(backbone(img.cuda()), labels.cuda())
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            amp.step(opt); amp.update(); opt.zero_grad(); lr_scheduler.step()
            
            if i % 200 == 0 and rank == 0:
                logging.info(f"E[{epoch}] S[{i}] Loss: {loss.item():.4f} Attn: {module_fc.last_attn:.3f} LR: {opt.param_groups[0]['lr']:.6f}")

        validate(backbone.module if world_size > 1 else backbone, cfg.rec, cfg.val_targets)
        if rank == 0: torch.save(backbone.module.state_dict() if world_size > 1 else backbone.state_dict(), os.path.join(cfg.output, "model.pt"))

def main():
    np.bool, np.int, np.float = np.bool_, np.int_, np.float64
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="dual_attention")
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    # Smart Path Finder
    possible_roots = [
        "/slurm/homes/bel/Atten dualFace Project /dali_emore",
        "/slurm/homes/bel/Atten dualFace Project/dali_emore",
        "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore",
        "/slurm/homes/bel/Atten dualFace Project/faces_emore/faces_emore"
    ]
    final_rec = next((p for p in possible_roots if os.path.exists(p)), None)
    if not final_rec: return print("[ERROR] Data not found.")

    rank, local_rank, world_size = setup_distributed()
    
    cfg = edict({
        'network': "r100", 'embedding_size': 512, 'batch_size': 256, 
        'lr': 0.1, 'num_epoch': 25, 'milestones': [10, 18, 22],
        'm1': 0.4, 'm2': 0.02, 'rec': final_rec,
        'val_targets': ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"],
        'output': f"./output/ablation_{args.attention}",
        'num_image': 5822653, 'seed': 2048, 'tensorboard': args.tensorboard
    })
    train(cfg, rank, local_rank, world_size)

if __name__ == "__main__":
    main()