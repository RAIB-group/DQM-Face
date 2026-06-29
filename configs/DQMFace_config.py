"""
DQM-Face Configuration
============================
Configuration settings for training DQM-Face on MS1MV2 (faces_emore).
"""

from easydict import EasyDict as edict

def get_config(attention_type="dqmface"):
    """
    Get configuration for DQM-Face training.
    Args:
        attention_type: 'dqmface' (Dual Quality Margin: Semantic + Magnitude)
    """
    config = edict()

    # ============== Model Settings ==============
    config.network = "r100"          # iResNet-100 (matches paper)
    config.embedding_size = 512
    config.fp16 = True

    # ============== Attention Settings ==============
    config.attention_type = attention_type
    config.attention_reduction = 64  # Bottleneck reduction for Semantic Quality

    # ============== Loss Settings ==============
    config.scale = 64.0
    config.m1 = 0.5          # Base positive margin (adaptive base)
    config.m2 = 0.1          # Negative margin (updated dynamically in train.py)

    # ============== Training Settings ==============
    config.optimizer = "sgd"
    config.lr = 0.1
    config.batch_size = 256  # Set to 512 if using 80GB A100/H100 GPUs
    config.num_epoch = 25    # Matches paper training schedule
    config.warmup_epoch = 1
    config.weight_decay = 5e-4
    config.momentum = 0.9
    config.gradient_acc = 1

    # ============== Dataset Settings ==============
    # Standard relative path for public repository
    config.rec = "datasets/faces_emore/faces_emore" 
    config.num_classes = 85742
    config.num_image = 5822653

    # ============== Validation Settings ==============
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cfp_ff", "cplfw"]

    # ============== Output Settings ==============
    config.output = f"output/DQM-Face_{attention_type}"
    config.save_all_states = True
    config.resume = False
    config.verbose = 2500
    config.frequent = 100

    # ============== Other Settings ==============
    config.seed = 2048
    config.num_workers = 8
    config.dali = False
    config.dali_aug = False
    config.sample_rate = 1.0
    config.using_wandb = False

    return config


# Pre-defined configs for easy importing
DQM_CONFIGS = {
    'dqmface': get_config('dqmface'),
    'baseline': get_config('baseline'), # Standard ArcFace if needed
}