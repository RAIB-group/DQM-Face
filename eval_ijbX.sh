#!/bin/bash -l
#SBATCH --partition=irb-20GB
#SBATCH --time=5-00:10:00
#SBATCH --output=/slurm/homes/%u/job_logs/jupyterhub_slurmspawner_%j.log
#SBATCH --job-name=face_train 
#SBATCH --export=ALL,HOME="/slurm/homes/bel/Atten dualFace Project /",SHELL=/bin/bash,PATH=/usr/lib/irb-jupyterhub/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/slurm/homes/bel/.local/bin
#SBATCH --mem=14000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

echo "Starting testing at $(date)"

export PYTHONWARNINGS="ignore"

/slurm/homes/bel/.conda/envs/pytorch-custom/bin/python -u "/slurm/homes/bel/Atten dualFace Project /insightface/recognition/arcface_torch/validate_lfw.py" \
    --repo-root "/slurm/homes/bel/Atten dualFace Project /insightface/recognition/arcface_torch/"\
    --weights "/slurm/homes/bel/Atten dualFace Project/output/output_ablation_dual_attention/20260130_170950/model.pt" \
    --data-dir "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore" \
    --network r100 \
    --val-targets lfw cfp_fp agedb_30 calfw cplfw cfp_ff
/slurm/homes/bel/.conda/envs/pytorch-custom/bin/python -u "/slurm/homes/bel/Atten dualFace Project /insightface/recognition/arcface_torch/eval_ijbc.py"\
 --model-prefix "/slurm/homes/bel/Atten dualFace Project/output/output_ablation_dual_attention/20260130_170950/model.pt" \
 --image-path "/slurm/homes/bel/Atten dualFace Project /ijb-testsuite/ijb/IJBC" \
 --result-dir "/slurm/homes/bel/Atten dualFace Project /output_ablation_dual/results" \
 --network r100 --batch-size 256 --job insightface --target IJBC

echo "Testing finished at $(date)"



