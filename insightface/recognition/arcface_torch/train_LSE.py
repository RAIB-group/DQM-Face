"""
Ablation Study Training Script: LSE Attention Contribution
================================================
Based on your working Baseline script.
"""

import argparse
import logging
import os
import json
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

# ============================================================
# ATTENTION MODULES
# ============================================================

class DualAttention(nn.Module):
    """BASELINE: Original simple MLP attention"""
    def __init__(self, in_features, **kwargs):
        super(DualAttention, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        raw_score = self.sigmoid(x)
        return 0.5 + raw_score

class LSEAttention(nn.Module):
    """
    NEW CONTRIBUTION: LSE (Log-Sum-Exp) Attention
    Acts as a 'Soft-Max' aggregator over multiple feature projections (facets).
    Helps in detecting the most prominent difficulty factors.
    """
    def __init__(self, in_features, num_facets=4, hidden_dim=128, **kwargs):
        super(LSEAttention, self).__init__()
        self.num_facets = num_facets
        # Project into multiple facets
        self.facets_fc = nn.Linear(in_features, hidden_dim * num_facets)
        self.relu = nn.PReLU()
        # Each facet produces a score
        self.score_fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.facets_fc(x)
        x = self.relu(x)
        
        # Reshape to [B, facets, hidden]
        x = x.view(batch_size, self.num_facets, -1)
        
        # Get scores for each facet: [B, facets, 1]
        facet_scores = self.score_fc(x)
        
        # Log-Sum-Exp over the facets dimension
        lse_score = torch.logsumexp(facet_scores, dim=1) 
        
        # Map to [0.5, 1.5] range
        atten_score = self.sigmoid(lse_score)
        return 0.5 + atten_score


ATTENTION_REGISTRY = {
    "dual_attention": DualAttention,
    "lse_attention": LSEAttention, # Registered new method
}


def get_attention_module(attention_type, in_features, reduction=32):
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown: {attention_type}. Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    return ATTENTION_REGISTRY[attention_type](in_features, reduction=reduction)


# ============================================================
# LOSS MODULE
# ============================================================

class AttenDualPartialFC(nn.Module):
    """Attention-based Dual Margin Loss"""
    def __init__(
        self,
        embedding_size,
        num_classes,
        scale=64.0,
        m1=0.5,
        m2=0.1,
        attention_type="dual_attention",
        reduction=32,
        sample_rate=1.0,
    ):
        super(AttenDualPartialFC, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.attention_type = attention_type

        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.attention_net = get_attention_module(attention_type, embedding_size, reduction)
        self.last_attention_score = None

        print(
            f"[AttenDualPartialFC] {attention_type} | Classes: {num_classes} | s={scale}, m1={m1}, m2={m2}"
        )
    def update_m2(self, new_m2):
        self.m2 = new_m2
        
    def forward(self, embeddings, labels):
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings_norm, weight_norm)

        atten_score = self.attention_net(embeddings)
        self.last_attention_score = atten_score.mean().item()

        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)

        m1_dynamic = self.m1 * atten_score

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        target_cosine = torch.cos(theta + m1_dynamic)
        non_target_cosine = torch.cos(theta - self.m2)

        output = (one_hot * target_cosine + (1.0 - one_hot) * non_target_cosine) * self.scale
        return F.cross_entropy(output, labels)

    def get_attention_score(self):
        return self.last_attention_score if self.last_attention_score else 0.0


# ============================================================
# SIMPLE VALIDATION (UNCHANGED)
# ============================================================

import pickle
import sklearn
from sklearn.model_selection import KFold

def load_bin(path, image_size=(112, 112)):
    try:
        with open(path, "rb") as f:
            bins, issame_list = pickle.load(f, encoding="bytes")
    except Exception:
        with open(path, "rb") as f:
            bins, issame_list = pickle.load(f, encoding="bytes")

    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)

    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        try:
            import cv2
            img = cv2.imdecode(np.frombuffer(_bin, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            import mxnet as mx
            img = mx.image.imdecode(_bin).asnumpy()

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = (img / 255.0 - 0.5) / 0.5
        for flip in [0, 1]:
            if flip == 1: img = torch.flip(img, [2])
            data_list[flip][idx] = img
    return data_list, issame_list

def evaluate_accuracy(embeddings, issame_list):
    embeddings1, embeddings2 = embeddings[0::2], embeddings[1::2]
    dist = np.sum(np.square(embeddings1 - embeddings2), 1)
    thresholds = np.arange(0, 4, 0.01)
    issame = np.array(issame_list)
    accuracy_list = [np.mean(np.less(dist, t) == issame) for t in thresholds]
    return max(accuracy_list), thresholds[np.argmax(accuracy_list)]

@torch.no_grad()
def validate(backbone, data_path, val_targets, batch_size=64):
    backbone.eval()
    results = {}
    for name in val_targets:
        bin_path = os.path.join(data_path, f"{name}.bin")
        if not os.path.exists(bin_path): continue
        try:
            data_list, issame_list = load_bin(bin_path)
            embeddings_list = []
            for data in data_list:
                embeddings = []
                for i in range(0, len(data), batch_size):
                    batch = data[i : i + batch_size].cuda(non_blocking=True)
                    emb = backbone(batch).cpu().numpy()
                    embeddings.append(emb)
                embeddings_list.append(np.concatenate(embeddings))
            embeddings = sklearn.preprocessing.normalize(embeddings_list[0] + embeddings_list[1])
            acc, _ = evaluate_accuracy(embeddings, issame_list)
            results[name] = acc
            print(f"  [Val] {name}: {acc*100:.2f}%")
        except Exception as e: print(f"  [Val] {name}: ERROR - {e}")
    backbone.train()
    return results


# ============================================================
# CONFIGURATION
# ============================================================

def get_config(attention_type, dataset_path=None):
    config = edict()
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.network = "r100"
    config.embedding_size = 512
    config.fp16 = True
    config.attention_type = attention_type
    config.attention_reduction = 64
    config.scale = 64.0
    config.m1 = 0.5
    config.m2 = 0.1
    config.optimizer = "sgd"
    config.lr = 0.1
    config.batch_size = 256
    config.num_epoch = 25
    config.milestones = [10, 18, 22]
    config.warmup_epoch = 0
    config.weight_decay = 5e-4
    config.momentum = 0.9
    config.gradient_acc = 1
    config.rec = dataset_path or "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    config.output = f"/slurm/homes/bel/Atten dualFace Project/output/output_ablation_{attention_type}/{date_str}"
    config.save_all_states = True
    config.resume = False
    config.verbose = 20000
    config.frequent = 200
    config.seed = 2048
    config.num_workers = 8
    config.dali = False
    config.dali_aug = False
    config.sample_rate = 1.0
    config.tensorboard = False
    config.tb_logdir = None
    config.tb_name = None
    return config

ABLATION_INFO = {
    "dual_attention": ("★ BASELINE", "Original simple MLP attention"),
    "lse_attention": ("★ NEW", "Log-Sum-Exp Facet Attention"),
}

# ============================================================
# TRAINING
# ============================================================

def setup_distributed():
    try:
        rank, local_rank, world_size = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    except KeyError:
        rank, local_rank, world_size = 0, 0, 1
        distributed.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def setup_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", 
                        handlers=[logging.FileHandler(os.path.join(output_dir, "training.log")), logging.StreamHandler()] if rank == 0 else [])

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def train(cfg, rank, local_rank, world_size):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); torch.cuda.set_device(local_rank)
    setup_logging(cfg.output, rank)

    marker, desc = ABLATION_INFO.get(cfg.attention_type, ("? ", "Unknown"))
    logging.info("=" * 70); logging.info(f"{marker} {cfg.attention_type.upper()} | {desc}"); logging.info("=" * 70)

    from backbones import get_model
    from dataset import get_dataloader

    train_loader = get_dataloader(cfg.rec, local_rank, cfg.batch_size, cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers)
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[local_rank], find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    module_fc = AttenDualPartialFC(embedding_size=cfg.embedding_size, num_classes=cfg.num_classes, 
                                   scale=cfg.scale, m1=cfg.m1, m2=cfg.m2, attention_type=cfg.attention_type).cuda()

    opt = torch.optim.SGD([{"params": backbone.parameters()}, {"params": module_fc.parameters()}],
                          lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    steps_per_epoch = cfg.num_image // (cfg.batch_size * world_size)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[m * steps_per_epoch for m in cfg.milestones], gamma=0.1)
    amp = torch.cuda.amp.GradScaler(growth_interval=100)

    loss_am, atten_am = AverageMeter(), AverageMeter()
    
    for epoch in range(cfg.num_epoch):
        module_fc.update_m2(0.05 if epoch < 10 else 0.1)
        if hasattr(train_loader, 'sampler'): train_loader.sampler.set_epoch(epoch)

        for img, labels in train_loader:
            embeddings = backbone(img)
            loss = module_fc(embeddings, labels)

            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            amp.step(opt); amp.update(); opt.zero_grad(); lr_scheduler.step()

            loss_am.update(loss.item()); atten_am.update(module_fc.get_attention_score())

            if (cfg.num_image // (cfg.batch_size * world_size) * epoch + 1) % cfg.frequent == 0:
                logging.info(f"[{cfg.attention_type}] E[{epoch}] Loss:{loss_am.avg:.4f} Attn:{atten_am.avg:.3f} LR:{opt.param_groups[0]['lr']:.6f}")

        if rank == 0:
            validate(backbone.module, cfg.rec, cfg.val_targets)
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, "model.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", type=str, default="lse_attention", choices=["dual_attention", "lse_attention"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    
    # NumPy compatibility fix
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64
    np.complex = np.complex128
    np.object = np.object_
    np.str = np.str_

    args = parser.parse_args()
    rank, local_rank, world_size = setup_distributed()
    cfg = get_config(args.attention, args.dataset)
    train(cfg, rank, local_rank, world_size)

if __name__ == "__main__":
    main()