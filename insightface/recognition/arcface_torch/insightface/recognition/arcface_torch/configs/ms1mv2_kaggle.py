
from easydict import EasyDict as edict

config = edict()
config.margin_list = (.0, 0.5, 0.0)
config.network = "r100"  # Options: r18, r34, r50, r100, mbf (MobileFaceNet)
config.resume = False
config.output = "Atten dualFace Project /output"
config.embedding_size = 512
config.sample_rate = .0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256  # Reduced for Kaggle GPU memory
config.lr = 0.1
config.verbose = 2500
config.dali = False
config.seed = 2048
config.num_workers = 8
config.gradient_acc = 1

# Dataset paths - UPDATE THESE to your Kaggle dataset paths
config.rec = "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore"  # Path to your MS1MV2 dataset
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 25
config.warmup_epoch = 1
config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw","cfp_ff","cplfw"]

# Additional settings
config.optimizer = "sgd"
config.frequent = 100
config.interclass_filtering_threshold = 0
config.save_all_states = True

# WandB (optional - set to False if not using)
config.using_wandb = False
config.wandb_key = ""
config.wandb_entity = ""
config.wandb_project = ""
config.wandb_resume = False
config.suffix_run_name = None
config.notes = ""
config.wandb_log_all = False
config.save_artifacts = False
