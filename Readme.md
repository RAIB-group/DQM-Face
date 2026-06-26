
# DQM-Face: Dual Quality Margin Learning for Face Recognition

## Authors

- El Ouanas Belabbaci
- Bhavesh Wani
- Philipp Terhörst

## Abstract

Face recognition in unconstrained environments remains highly challenging due to diverse and extreme variations encountered in real-world scenarios. To mitigate these effects, existing margin-based approaches model sample quality through feature magnitude. However, magnitude-based modeling alone is susceptible to identity-agnostic noise, which can degrade the reliability and discriminative power of learned representations. In this paper, we propose Dual Quality Margin Learning for Face Recognition (DQM-Face), a novel framework that enables refined attraction and repulsion dynamics during representation learning. Our approach unifies conventional magnitude-based quality estimation with a newly introduced semantic quality learning mechanism, realized via squeeze-and-excitation semantic attention. By jointly leveraging magnitude and semantic cues, we construct enhanced quality-aware margins that adaptively strengthen intra-class compactness through improved attraction during learning. To further enhance inter-class discrimination, we introduce a repulsion margin formulation that explicitly enlarges inter-class separation. The unified integration of semantic quality modeling with dual attraction–repulsion margin optimization results in a more structured and discriminative feature geometry. Extensive experiments on multiple challenging benchmarks demonstrate that DQM-Face consistently outperforms state-of-the-art face recognition methods. Moreover, we show that the quality learned for margin optimization is highly effective for face image quality assessment within the proposed framework, demonstrating that the learned quality signal is intrinsically aligned with the recognition objective.


## Installation Setup

1. Add the face_emore dataset to the DQM-Face directory.
2. Add the IJB-B and IJB-C datasets to the following directory:
   `ijb-testsuite/ijb`
3. The training model code is located at:
   update the local path in `insightface/recognition/arcface_torch/train_NATTX2.py`
4. Use the following scripts to run the training code. Make sure to update the paths as needed:

   - `run_train80G.sh`
   - `run_train40G.sh`
   - `run_train20G.sh`

5. After training is completed, use the following scripts to evaluate the model on IJB-B and IJB-C:

   - `eval_ijbX.sh`
   - `eval_ijb.sh`

6. The trained model will be saved in the DQM-Face folder, with the filename including the date and time, for example:

   ```text
   2026-02-01 06:09:15,809 Epoch 24 done | Loss: 12.8672 | Attn: 0.011
   2026-02-01 06:09:18,960 Done! Model: output/output_ablation_dual_attention/20260130_170950/model.pt
   Training finished at Sun Feb  1 06:09:26 AM CET 2026
   ```

Note:
You must update all dataset and directory paths in the scripts and code to match your local setup.

## Acknowledgement

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant 544631027.