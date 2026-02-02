"""
SOTA Training Script: Magnitude-Aware Adaptive Margin
Backbone: iResNet-100
Benchmarking: LFW, CFP-FP, AgeDB-30, CALFW, CPLFW
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

class SOTAQualityAttention(nn.Module):
    def __init__(self, in_features, l_a=10.0, u_a=110.0, **kwargs):
        super().__init__()
        self.l_a, self.u_a = l_a, u_a
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        mag_q = (torch.clamp(norm, self.l_a, self.u_a) - self.l_a) / (self.u_a - self.l_a)
        sem_q = self.trunk(x)
        # Range is 0.5 to 1.5 to match your original BASELINE style
        combined_score = 0.5 + (0.5 * mag_q + 0.5 * sem_q) 
        return combined_score, norm

ATTENTION_REGISTRY = {
    "dual_attention": SOTAQualityAttention, 
}

def get_attention_module(attention_type, in_features, reduction=32):
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown: {attention_type}")
    return ATTENTION_REGISTRY[attention_type](in_features)

# ============================================================
# LOSS MODULE
# ============================================================

class AttenDualPartialFC(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=64.0, m1=0.5, m2=0.1,
                 attention_type="dual_attention", reduction=32, sample_rate=1.0):
        super(AttenDualPartialFC, self).__init__()
        self.scale, self.m1, self.m2 = scale, m1, m2
        self.u_a = 110.0
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.attention_net = get_attention_module(attention_type, embedding_size, reduction)
        self.last_attention_score = None

        # Exact print from your original code
        print(f"[AttenDualPartialFC] {attention_type} | Classes: {num_classes} | s={scale}, m1={m1}, m2={m2}")

    def update_m2(self, new_m2):
        self.m2 = new_m2
        
    def forward(self, embeddings, labels):
        atten_score, norm = self.attention_net(embeddings)
        self.last_attention_score = atten_score.mean().item()

        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings_norm, weight_norm)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # MagFace dynamic margin logic
        m1_dynamic = self.m1 * atten_score

        theta = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        target_cosine = torch.cos(theta + m1_dynamic)
        non_target_cosine = torch.cos(theta - self.m2)

        output = one_hot * target_cosine + (1.0 - one_hot) * non_target_cosine
        output *= self.scale

        # SOTA Regularization (MagFace)
        reg_mag = torch.mean(1.0 / (norm + 1e-4) + (norm / self.u_a))
        return F.cross_entropy(output, labels) + (35.0 * reg_mag)

    def get_attention_score(self):
        return self.last_attention_score if self.last_attention_score else 0.0
# ============================================================
# CONFIGURATION
# ============================================================



def get_config(attention_type, dataset_path=None):
    config = edict()
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config.network = "r100"       
    config.embedding_size = 512
    config.fp16 = True

    config.attention_type = attention_type
    config.attention_reduction = 64

    config.scale = 64.0
    config.m2 = 0.05              
    config.l_a = 10.0
    config.u_a = 110.0
    
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
    config.dali_aug = False  # FIX: Defined here to prevent AttributeError
    config.sample_rate = 1.0

    config.tensorboard = True
    config.tb_logdir = None
    config.tb_name = None

    return config


ABLATION_INFO = {
    "dual_attention": ("★ SOTA-UPGRADE", "Magnitude-Aware Residual Attention (MagFace/QMagFace Principle)"),
}

# ============================================================
# TRAINING
# ============================================================

def setup_distributed():
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("gloo")
    except KeyError:
        rank = 0
        local_rank = 0
        world_size = 1
        distributed.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )
    return rank, local_rank, world_size


def setup_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
        if rank == 0
        else [logging.FileHandler(log_file)],
    )


def setup_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(cfg, rank, local_rank, world_size):
    """Main training function"""
    setup_seed(cfg.seed)
    torch.cuda.set_device(local_rank)
    setup_logging(cfg.output, rank)

    marker, desc = ABLATION_INFO.get(cfg.attention_type, ("? ", "Unknown"))

    logging.info("=" * 70)
    logging.info(f"{marker} {cfg.attention_type.upper()}")
    logging.info(f"   {desc}")
    logging.info("=" * 70)

    # TensorBoard (rank 0 only)
    writer = None
    if getattr(cfg, "tensorboard", False) and rank == 0:
        tb_root = cfg.tb_logdir or os.path.join(cfg.output, "tb")
        run_name = cfg.tb_name or f"{cfg.attention_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_dir = os.path.join(tb_root, run_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        logging.info(f"TensorBoard enabled. Log dir: {tb_dir}")

        # Save hparams-ish info
        try:
            writer.add_text("run/attention_type", str(cfg.attention_type), 0)
            writer.add_text("run/network", str(cfg.network), 0)
            writer.add_text("run/output", str(cfg.output), 0)
        except Exception:
            pass

    # Import here to avoid any import-time issues in some environments
    from backbones import get_model
    from dataset import get_dataloader
    from lr_scheduler import PolynomialLRWarmup

    # Data
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers,
    )

    # Model
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        backbone,
        broadcast_buffers=False,
        device_ids=[local_rank],
        bucket_cap_mb=16,
        find_unused_parameters=True,
    )
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    backbone._set_static_graph()

    logging.info(
        f"Backbone: {cfg.network}, Params: {sum(p.numel() for p in backbone.parameters()):,}"
    )

    # Loss
    module_fc = AttenDualPartialFC(
        embedding_size=cfg.embedding_size,
        num_classes=cfg.num_classes,
        scale=cfg.scale,
        m1=cfg.m1,
        m2=cfg.m2,
        attention_type=cfg.attention_type,
        reduction=cfg.attention_reduction,
    ).cuda()
    module_fc.train()

    # Optimizer
    opt = torch.optim.SGD(
        [{"params": backbone.parameters()}, {"params": module_fc.parameters()}],
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler
    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

  # --- OLD CODE TO REMOVE ---
# lr_scheduler = PolynomialLRWarmup(opt, cfg.warmup_step, cfg.total_step)
# Scheduler
    cfg.total_batch_size = cfg.batch_size * world_size
    steps_per_epoch = cfg.num_image // cfg.total_batch_size
    milestones_steps = [m * steps_per_epoch for m in cfg.milestones]

    # Using MultiStepLR as requested
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, 
        milestones=milestones_steps, 
        gamma=0.1
    )

    # Fixed Indentation here
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        ckpt = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
        if os.path.exists(ckpt):
            checkpoint = torch.load(ckpt, map_location="cpu")
            start_epoch = checkpoint["epoch"]
            global_step = checkpoint["global_step"]
            backbone.module.load_state_dict(checkpoint["state_dict_backbone"])
            module_fc.load_state_dict(checkpoint["state_dict_fc"])
            opt.load_state_dict(checkpoint["state_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["state_lr_scheduler"])
            logging.info(f"Resumed from epoch {start_epoch}")

    # Training
    loss_am = AverageMeter()
    atten_am = AverageMeter()
    amp = torch.cuda.amp.GradScaler(growth_interval=100)

    results = {"attention_type": cfg.attention_type, "training": [], "validation": {}}

    logging.info(f"Training: {cfg.num_epoch} epochs, {cfg.total_step} steps")

    for epoch in range(start_epoch, cfg.num_epoch):
        if epoch < 10:
            current_m2 = 0.05
        elif epoch < 18:
            current_m2 = 0.1
        else:
            current_m2 = 0.1
        
        # Apply the update
        module_fc.update_m2(current_m2)
        logging.info(f"Epoch {epoch}: Inter-class margin m2 set to {current_m2}")
 
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for img, labels in train_loader:
            global_step += 1

            embeddings = backbone(img)
            loss = module_fc(embeddings, labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()

            lr_scheduler.step()

            loss_am.update(loss.item(), 1)
            atten_am.update(module_fc.get_attention_score(), 1)

            if global_step % cfg.frequent == 0:
                attn = module_fc.get_attention_score()
                lr = opt.param_groups[0]['lr']
                
                # RESTORED TO YOUR EXACT ORIGINAL STYLE
                logging.info(
                    f"★ BASELINE [{cfg.attention_type}] "
                    f"E[{epoch}/{cfg.num_epoch}] S[{global_step}] "
                    f"Loss:{loss.item():.4f} Attn:{attn:.3f} m2:{module_fc.m2} LR:{lr:.6f}"
                )

                if writer is not None:
                    writer.add_scalar("train/loss", loss_am.avg, global_step)
                    writer.add_scalar("train/attention", atten_am.avg, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

            if global_step % cfg.verbose == 0 and global_step > 0 and rank == 0:
                logging.info("Running validation...")
                val_results = validate(backbone.module, cfg.rec, cfg.val_targets)
                results["validation"][global_step] = val_results

                if writer is not None:
                    for ds_name, acc in val_results.items():
                        writer.add_scalar(f"val/{ds_name}", float(acc), global_step)
                    if len(val_results) > 0:
                        writer.add_scalar(
                            "val/mean_acc",
                            float(np.mean(list(val_results.values()))),
                            global_step,
                        )
        logging.info("Running Epoch validation...")
        val_results = validate(backbone.module, cfg.rec, cfg.val_targets)
        results["validation"][global_step] = val_results

        if writer is not None:
            for ds_name, acc in val_results.items():
                writer.add_scalar(f"val/{ds_name}", float(acc), global_step)
            if len(val_results) > 0:
                writer.add_scalar(
                    "val/mean_acc",
                    float(np.mean(list(val_results.values()))),
                    global_step,
                    )

        # Epoch end
        results["training"].append(
            {"epoch": epoch, "loss": loss_am.avg, "attention": atten_am.avg}
        )

        logging.info(f"Epoch {epoch} done | Loss: {loss_am.avg:.4f} | Attn: {atten_am.avg:.3f}")

        if writer is not None:
            writer.add_scalar("epoch/loss", loss_am.avg, epoch)
            writer.add_scalar("epoch/attention", atten_am.avg, epoch)

        # Save
        if cfg.save_all_states:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "state_dict_backbone": backbone.module.state_dict(),
                    "state_dict_fc": module_fc.state_dict(),
                    "state_optimizer": opt.state_dict(),
                    "state_lr_scheduler": lr_scheduler.state_dict(),
                },
                os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"),
            )

        if rank == 0:
            torch.save(backbone.module.state_dict(), os.path.join(cfg.output, "model.pt"))

        if cfg.dali:
            train_loader.reset()

    # Final
    if rank == 0:
        torch.save(backbone.module.state_dict(), os.path.join(cfg.output, "model.pt"))
        with open(os.path.join(cfg.output, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Done! Model: {cfg.output}/model.pt")

    if writer is not None:
        writer.flush()
        writer.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Training")
    parser.add_argument(
        "--attention",
        type=str,
        default="dual_attention",
        choices=["dual_attention"],
    )
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--resume", action="store_true")

    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging (rank 0 only)")
    parser.add_argument("--tb-logdir", type=str, default=None, help="TensorBoard root logdir (default: <output>/tb)")
    parser.add_argument("--tb-name", type=str, default=None, help="Optional run name (subfolder)")

    # NumPy fix
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64
    np.complex = np.complex128
    np.object = np.object_
    np.str = np.str_

    args = parser.parse_args()

    if args.list:
        print("\nAvailable configurations:")
        print("=" * 50)
        for name, (marker, desc) in ABLATION_INFO.items():
            print(f"  {marker} {name}: {desc}")
        print("=" * 50)
        return

    rank, local_rank, world_size = setup_distributed()
    torch.backends.cudnn.benchmark = True

    if args.run_all:
        all_results = {}
        for att_type in ["dual_attention"]:
            print(f"\n{'='*70}")
            print(f"Training: {att_type}")
            print(f"{'='*70}")
            cfg = get_config(att_type, args.dataset)
            cfg.tensorboard = args.tensorboard
            cfg.tb_logdir = args.tb_logdir
            cfg.tb_name = args.tb_name
            all_results[att_type] = train(cfg, rank, local_rank, world_size)

        if rank == 0:
            out_path = os.path.join("output", "ablation_all_results.json")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print("\n✓ All ablations complete!")
            print(f"Results saved to: {out_path}")
    else:
        cfg = get_config(args.attention, args.dataset)
        if args.resume:
            cfg.resume = True

        cfg.tensorboard = args.tensorboard
        cfg.tb_logdir = args.tb_logdir
        cfg.tb_name = args.tb_name

        train(cfg, rank, local_rank, world_size)


if __name__ == "__main__":
    main()