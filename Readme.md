# Dual Quality Margin Learning for Face Recognition

_Accepted at the European Conference on Computer Vision (ECCV) 2026._

## DQM-Face: Learning to Attract and Repel: Dual Quality Margin Learning for Face Recognition

* [Research Paper](#) *(Link coming soon)*

## Table of Contents 

- [Abstract](#abstract)
- [Results & Models](#results--models)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

<img src="assets/fig_marge.png" width="500" align="right">

Face recognition systems in unconstrained environments have to deal with extreme variations (such as pose, illumination, and occlusion). To mitigate these effects, existing margin-based approaches model sample quality through feature magnitude. However, magnitude-based modeling alone is susceptible to identity-agnostic noise, which can degrade the reliability and discriminative power of learned representations. 

In this work, we propose **Dual Quality Margin Learning for Face Recognition (DQM-Face)**, a novel framework that enables refined attraction and repulsion dynamics during representation learning. Our approach unifies conventional magnitude-based quality estimation with a newly introduced semantic quality learning mechanism, realized via squeeze-and-excitation semantic attention. By jointly leveraging magnitude and semantic cues, we construct enhanced quality-aware margins that adaptively strengthen intra-class compactness through improved attraction during learning. To further enhance inter-class discrimination, we introduce a repulsion margin formulation that explicitly enlarges inter-class separation. 

The unified integration of semantic quality modeling with dual attraction–repulsion margin optimization results in a more structured and discriminative feature geometry. Extensive experiments demonstrate that DQM-Face consistently outperforms state-of-the-art face recognition methods on multiple challenging benchmarks.

<br clear="all">

## Results & Models

The proposed approach is analyzed in three steps. First, we provide our pre-trained models. Second, we visually demonstrate how our dual-quality mechanism focuses on identity-preserving features. Third, we evaluate the learned quality for Face Image Quality Assessment (FIQA) to show its effectiveness in rejecting low-utility samples. For comprehensive recognition benchmark tables (IJB-B, IJB-C, LFW, AgeDB, etc.), please refer to our paper.

### Pre-trained Models
We provide pre-trained models based on the iResNet-100 backbone trained on the MS1MV2 dataset. You can download them here:
* [**DQM-Face (α = 0.5)**](https://drive.google.com/file/d/1V1zmSWtPx7jKI4fQ-LzUPWTIrwjkPrik/view?usp=sharing) - Best overall model (Magnitude + Semantic fusion).
* [**DQM-Face (α = 0.4)**](https://drive.google.com/file/d/1kC_HitTbsHwlIcwfXZ0UAUO7ce3oun3P/view?usp=sharing) - Alternate fusion weighting.
* [**DQM-Face qsem (α = 1.0)**](https://drive.google.com/file/d/1aWGbFbMGMAlg9GUYHfOXQ4RvqzPi8HJp/view?usp=sharing) - Semantic quality only (without magnitude).

<br>

<img src="assets/gradcam.png" width="550" align="right">

**Visual Attribution Analysis (Grad-CAM)** - To provide qualitative insight into the different quality branches, we employ Grad-CAM to visualize the spatial regions that contribute most to the recognition decision. The magnitude-only variant often exhibits attention dispersed toward non-discriminative regions (e.g., background textures) when affected by blur or occlusion. In contrast, our proposed DQM-Face (fused quality, α = 0.5) combines the strengths of both quality cues, exhibiting well-localized and stable activation over the most informative facial regions while remaining robust to challenging imaging conditions.

<br clear="all">
<br>

<table align="right" style="width:450px;">
  <tr>
    <td><img src="assets/lfw_evr.png" alt="LFW"></td>
    <td><img src="assets/adience_evr.png" alt="Adience"></td>
  </tr>
  <tr>
    <td><img src="assets/cplfw_evr.png" alt="CPLFW"></td>
    <td><img src="assets/xqlfw_evr.png" alt="XQLFW"></td>
  </tr>
</table>

**Face Image Quality Assessment (FIQA) Performance** - To measure the effectiveness of the proposed quality estimate utilized for margin learning, we evaluate FIQA performance using Error-vs-Discard (EvD) characteristics. The DQM-Face model demonstrates consistently strong performance, ranking among the top-performing methods across challenging datasets featuring age variations, pose variations, and unconstrained environments. This demonstrates that the learned quality signal is intrinsically aligned with the recognition objective, successfully learning identity-aware quality representations.

<br clear="all">

## Face Recognition Model Training

**Training Dataset:** In our paper, we employ the MS1MV2 dataset for training. This dataset can be downloaded from the InsightFace DataZoo (MS1M-ArcFace) via their [official datasets page](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_). Please ensure you strictly follow their license and distribution guidelines.

1. Download and unzip the dataset, then place it in your local `datasets/` folder.
2. Update the dataset path in `train.py` to point to this directory.
3. Run the training script using your preferred bash file (e.g., `bash scripts/run_train80G.sh`).

*Note: All code provided in this repository has been trained and tested using PyTorch 1.7.1.*

## Face Recognition Evaluation

**Evaluation on LFW, AgeDB-30, CPLFW, and CFP-FP:**

You can download the data from their official webpages. 
*Alternative:* The evaluation datasets are already available in the training dataset package as `.bin` files.

1. Set `config.rec` to your dataset folder (e.g., `datasets/faces_emore`).
2. Set `config.val_targets` to the list of the evaluation datasets you wish to test.
3. Download the pre-trained model from the links provided in the table above.
4. Set `config.output` to the path of the downloaded pre-trained model weights.
5. Run the evaluation script:
   ```bash
   python eval/evaluation.py

   
## FIQA evaluation

1. text text text text text text text text .
2. text text text text text text text text .
3. text text text text text text text text .

## Acknowledgement

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German
Research Foundation) under Grant 544631027.

## License

This project is licensed under the terms of the Attribution-NonCommercial 4.0
International (CC BY-NC 4.0) license. Copyright (c) 2026 Johannes Gutenberg
University Mainz (JGU). You are free to use, modify, and redistribute this
software for non-commercial research purposes, provided appropriate attribution
is given. Commercial use requires prior permission from the copyright holder.

