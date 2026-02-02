
1. Add the face_emore dataset to the FRS directory.
2. Add the IJB-B and IJB-C datasets to the following directory:
   `FRS/ijb-testsuite/ijb`
3. The best training model configuration is located at:
   `FRS/insightface/recognition/arcface_torch/train_NATTX2.py`
4. Use the following scripts to run the training code (make sure to update the paths as needed):

   * `run_train80G.sh`
   * `run_train40G.sh`
   * `run_train20G.sh`
5. After training is completed, use the following scripts to evaluate the model on IJB-B and IJB-C:

   * `eval_ijbX.sh`
   * `eval_ijb.sh`
6. The trained model will be saved in the FRS folder, with the filename including the date and time like this """2026-02-01 06:09:15,809 Epoch 24 done | Loss: 12.8672 | Attn: 0.011
2026-02-01 06:09:18,960 Done! Model: /slurm/homes/bel/FRS/output/output_ablation_dual_attention/20260130_170950/model.pt
Training finished at Sun Feb  1 06:09:26 AM CET 2026""". (from the job logs)


Note:
You must update all dataset and directory paths in the scripts and code to match your local setup.



==========================
Training Part

1) install requirment.txt DualAttFace/insightface/recognition/arcface_torch
2) Change path in /home/bw/FIQA/research_fr_fiqa/DualAttFace/insightface/recognition/arcface_torch/train_NATTX2.py 
    config.output = f"/home/bw/FIQA/research_fr_fiqa/DualAttFace/output/output_ablation_{attention_type}/{date_str}"
    config.rec = dataset_path or "/home/bw/FIQA/MS1MV2"

3) Change m2 value - current results need to be tested on 0.2 and so on... (note: higher the better)
    
    current best on (Here: can be changed)
        if epoch < 10:
            current_m2 = 0.05
        elif epoch < 18:
            current_m2 = 0.1
        else:
            current_m2 = 0.15

class AttenDualPartialFC(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=64.0, m1=0.5, m2=0.2,

4) Alpha - (0.4 best)
# Weighted fusion (Adaptive quality) [(1-alpha)mag + alpha*sem]
        combined_quality = 0.6 * mag_quality + 0.4 * sem_quality

5) m1 is fixed 

Scope of Improvements:
1) more epoch e.g. 40 (rn tested on 25) + lr
2) PowerMagAttention - Architect 


========================================
Evaluation