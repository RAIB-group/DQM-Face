
"""
Ablation Study Configuration
============================
Run different attention mechanisms for comparison
"""

from easydict import EasyDict as edict

def get_ablation_config(attention_type="hybrid"):
    """
    Get config for ablation study
    Args:
        attention_type: One of: 
            - 'baseline': No attention (standard margins)
            - 'dual_attention':  Original simple attention
            - 'uncertainty': Uncertainty-based attention
            - 'channel':  Channel attention (SE-style)
            - 'hybrid': Channel + Uncertainty (full model)
    """
    config = edict()

    # ============== Model Settings ==============
    config.network = "r50"
    config.embedding_size = 512
    config.fp16 = True

    # ============== Attention Settings ==============
    config.attention_type = attention_type
    config.attention_reduction = 32  # For channel/hybrid attention

    # ============== Loss Settings ==============
    config.scale = 64.0
    config.m1 = 0.5          # Base positive margin
    config.m2 = 0.1          # Negative margin

    # ============== Training Settings ==============
    config.optimizer = "sgd"
    config.lr = 0.1
    config.batch_size = 512
    config.num_epoch = 20
    config.warmup_epoch = 1
    config.weight_decay = 5e-4
    config.momentum = 0.9
    config.gradient_acc = 1

    # ============== Dataset Settings ==============
    # UPDATED PATH: Go up 3 levels to find the data
    config.rec = "faces_emore/faces_emore" 
    config.num_classes = 85742
    config.num_image = 5822653

    # ============== Validation Settings ==============
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw","cfp_ff","cplfw"]

    # ============== Output Settings ==============
    # UPDATED PATH: Go up 3 levels to find the output folder
    config.output = f"output/output_{attention_type}"
    config.save_all_states = True
    config.resume = False
    config.verbose = 2000
    config.frequent = 100

    # ============== Other Settings ==============
    config.seed = 2048
    config.num_workers = 4
    config.dali = False
    config.dali_aug = False
    config.sample_rate = 1.0
    config.using_wandb = False

    return config


# Pre-defined configs for each ablation
ABLATION_CONFIGS = {
    'baseline': get_ablation_config('baseline'),
    'dual_attention': get_ablation_config('dual_attention'),
    'uncertainty':  get_ablation_config('uncertainty'),
    'channel': get_ablation_config('channel'),
    'hybrid': get_ablation_config('hybrid'),
}
