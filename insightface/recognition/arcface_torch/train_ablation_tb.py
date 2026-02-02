"""
Ablation Study Training Script (Self-Contained)
================================================
Uses TensorBoard via torch.utils.tensorboard (no TensorFlow dependency).

BASELINE:  dual_attention (original AttentionBlock)
ABLATIONS:
  - uncertainty:  UncertaintyAttention
  - channel:  ChannelAttentionBlock
  - hybrid: HybridAttentionBlock (Full Model)
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


class UncertaintyAttention(nn.Module):
    """ABLATION: Log-variance uncertainty estimation"""
    def __init__(self, in_features, **kwargs):
        super(UncertaintyAttention, self).__init__()
        self.fc_log_var = nn.Linear(in_features, 1)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        log_var = self.fc_log_var(x)
        log_var = self.bn(log_var)
        sigma = torch.exp(0.5 * log_var)
        atten_score = 1.0 / (sigma + 0.1)
        atten_score = torch.clamp(atten_score, 0.5, 1.5)
        return atten_score


class ChannelAttention(nn.Module):
    """ABLATION:  Squeeze-Excitation style attention"""
    def __init__(self, in_features, reduction=32, **kwargs):
        super(ChannelAttention, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )
        self.final_score = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        importance = self.se(x)
        x_refined = x * importance
        out = self.final_score(x_refined)
        raw_score = self.sigmoid(out)
        return 0.5 + raw_score


class HybridAttention(nn.Module):
    """FULL MODEL: Channel + Uncertainty"""
    def __init__(self, in_features, reduction=64, **kwargs):
        super(HybridAttention, self).__init__()
        self.channel_fc1 = nn.Linear(in_features, in_features // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc2 = nn.Linear(in_features // reduction, in_features, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.uncertainty_fc = nn.Linear(in_features, 1)
        self.uncertainty_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        weights = self.channel_fc1(x)
        weights = self.relu(weights)
        weights = self.channel_fc2(weights)
        weights = self.sigmoid(weights)
        x_refined = x * weights

        log_var = self.uncertainty_fc(x_refined)
        log_var = self.uncertainty_bn(log_var)
        sigma = torch.exp(0.5 * log_var)

        raw_score = 1.0 / (1.0 + sigma)
        return 0.5 + raw_score


ATTENTION_REGISTRY = {
    "dual_attention": DualAttention,
    "uncertainty": UncertaintyAttention,
    "channel": ChannelAttention,
    "hybrid": HybridAttention,
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
        m2=0,
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

        target_angle = theta + m1_dynamic
        target_cosine = torch.cos(target_angle)

        non_target_angle = theta - self.m2
        non_target_cosine = torch.cos(non_target_angle)

        output = one_hot * target_cosine + (1.0 - one_hot) * non_target_cosine
        output = output * self.scale

        return F.cross_entropy(output, labels)

    def get_attention_score(self):
        return self.last_attention_score if self.last_attention_score else 0.0


# ============================================================
# SIMPLE VALIDATION
# ============================================================

import pickle
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


def load_bin(path, image_size=(112, 112)):
    """Load validation bin file"""
    try:
        import mxnet as mx
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
            import mxnet as mx
            img = mx.image.imdecode(_bin).asnumpy()
        except Exception:
            import cv2
            img = cv2.imdecode(np.frombuffer(_bin, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = (img / 255.0 - 0.5) / 0.5

        for flip in [0, 1]:
            if flip == 1:
                img = torch.flip(img, [2])
            data_list[flip][idx] = img

    return data_list, issame_list


def evaluate_accuracy(embeddings, issame_list, nrof_folds=10):
    """Calculate accuracy using simple threshold sweep"""
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), 1)

    thresholds = np.arange(0, 4, 0.01)
    issame = np.array(issame_list)

    accuracy_list = []
    for threshold in thresholds:
        predict_issame = np.less(dist, threshold)
        accuracy = np.mean(predict_issame == issame)
        accuracy_list.append(accuracy)

    best_acc = max(accuracy_list)
    best_threshold = thresholds[np.argmax(accuracy_list)]

    return best_acc, best_threshold


@torch.no_grad()
def validate(backbone, data_path, val_targets, batch_size=64):
    """Run validation on targets"""
    backbone.eval()
    results = {}

    for name in val_targets:
        bin_path = os.path.join(data_path, f"{name}.bin")
        if not os.path.exists(bin_path):
            print(f"  [Val] {name}: NOT FOUND")
            continue

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

            embeddings = embeddings_list[0] + embeddings_list[1]
            embeddings = sklearn.preprocessing.normalize(embeddings)

            acc, threshold = evaluate_accuracy(embeddings, issame_list)
            results[name] = acc
            print(f"  [Val] {name}: {acc*100:.2f}%")
        except Exception as e:
            print(f"  [Val] {name}: ERROR - {e}")

    backbone.train()
    return results


# ============================================================
# CONFIGURATION
# ============================================================

def get_config(attention_type, dataset_path=None):
    config = edict()

    config.network = "r100"
    config.embedding_size = 512
    config.fp16 = True

    config.attention_type = attention_type
    config.attention_reduction = 64

    config.scale = 64.0
    config.m1 = 0.5
    config.m2 = 0

    config.optimizer = "sgd"
    config.lr = 0.1
    config.batch_size = 256

    config.num_epoch = 25
    config.milestones = [10, 18, 22]
    config.warmup_epoch = 1
    config.weight_decay = 5e-4
    config.momentum = 0.9
    config.gradient_acc = 1

    config.rec = dataset_path or "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653

    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]

    config.output = f"/slurm/homes/bel/Atten dualFace Project /output/ablation_{attention_type}"
    config.save_all_states = True
    config.resume = False
    config.verbose = 6000
    config.frequent = 200

    config.seed = 2048
    config.num_workers = 8
    config.dali = False
    config.dali_aug = False
    config.sample_rate = 1.0

    # TensorBoard
    config.tensorboard = False
    config.tb_logdir = None
    config.tb_name = None

    return config


ABLATION_INFO = {
    "dual_attention": ("★ BASELINE", "Original simple MLP attention"),
    "uncertainty": ("◆ ABLATION", "Log-variance uncertainty estimation"),
    "channel": ("◆ ABLATION", "Squeeze-Excitation feature refinement"),
    "hybrid": ("◆ FULL MODEL", "Channel + Uncertainty combined"),
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

    lr_scheduler = PolynomialLRWarmup(opt, cfg.warmup_step, cfg.total_step)

    # Resume
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
                lr = lr_scheduler.get_last_lr()[0]
                logging.info(
                    f"{marker} [{cfg.attention_type}] "
                    f"E[{epoch}/{cfg.num_epoch}] S[{global_step}/{cfg.total_step}] "
                    f"Loss:{loss_am.avg:.4f} Attn:{atten_am.avg:.3f} LR:{lr:.6f}"
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
        choices=["dual_attention", "uncertainty", "channel", "hybrid"],
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
        for att_type in ["dual_attention", "uncertainty", "channel", "hybrid"]:
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