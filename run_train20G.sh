#!/bin/bash -l
#SBATCH --partition=irb-20GB
#SBATCH --time=5-00:10:00
#SBATCH --output=/slurm/homes/%u/job_logs/jupyterhub_slurmspawner_%j.log
#SBATCH --job-name=face_train 
#SBATCH --export=ALL,HOME="/slurm/homes/bel/Atten dualFace Project /",SHELL=/bin/bash,PATH=/usr/lib/irb-jupyterhub/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/slurm/homes/bel/.local/bin
#SBATCH --mem=14000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1



conda activate pytorch-custom
cd "/slurm/homes/bel/Atten dualFace Project /insightface/recognition/arcface_torch"

echo "Starting training at $(date)"
export PYTHONWARNINGS="ignore"
/slurm/homes/bel/.conda/envs/pytorch-custom/bin/python -u train_NATTX2.py --attention dual_attention --tensorboard
echo "Training finished at $(date)"








