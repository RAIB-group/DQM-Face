# DQM-Face: Dual Quality Margin Learning for Face Recognition

**License:** CC BY-NC 4.0
**Framework:** PyTorch

Official PyTorch implementation of the paper:

**Learning to Attract and Repel: Dual Quality Margin Learning for Face Recognition (DQM-Face)**

**🎉 Accepted at ECCV 2026**

---

## 📖 Abstract

Face recognition in unconstrained environments remains highly challenging due to diverse and extreme variations encountered in real-world scenarios. To mitigate these effects, existing margin-based approaches model sample quality through feature magnitude. However, magnitude-based modeling alone is susceptible to identity-agnostic noise, which can degrade the reliability and discriminative power of learned representations.

In this paper, we propose **Dual Quality Margin Learning for Face Recognition (DQM-Face)**, a novel framework that enables refined attraction and repulsion dynamics during representation learning. Our approach unifies conventional magnitude-based quality estimation with a newly introduced semantic quality learning mechanism, realized via squeeze-and-excitation semantic attention. By jointly leveraging magnitude and semantic cues, we construct enhanced quality-aware margins that adaptively strengthen intra-class compactness through improved attraction during learning. To further enhance inter-class discrimination, we introduce a repulsion margin formulation that explicitly enlarges inter-class separation.

The unified integration of semantic quality modeling with dual attraction–repulsion margin optimization results in a more structured and discriminative feature geometry. Extensive experiments on multiple challenging benchmarks demonstrate that DQM-Face consistently outperforms state-of-the-art face recognition methods. Moreover, we show that the quality learned for margin optimization is highly effective for face image quality assessment within the proposed framework, demonstrating that the learned quality signal is intrinsically aligned with the recognition objective.

---

# 🚀 Installation & Data Preparation

## 1. Clone the repository

```bash
git clone https://github.com/RAIB-group/DQM-Face.git
cd DQM-Face
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare the training dataset

Download the **MS1MV2 (faces_emore)** dataset and place it under:

```text
datasets/
└── faces_emore/
```

> **Note:** Update all dataset and output paths in `configs/ablation_config.py` to match your local environment.

---

# 📝 Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{belabbaci2026dqmface,
  title={Learning to Attract and Repel: Dual Quality Margin Learning for Face Recognition (DQM-Face)},
  author={Belabbaci, El Ouanas and Wani, Bhavesh and Terh{\"o}rst, Philipp},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

# 🤝 Acknowledgement

This work was funded by the **Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)** under Grant **544631027**.

---

# ⚖️ License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** License.

Copyright © 2026 **Johannes Gutenberg University Mainz (JGU)**.

You are free to use, modify, and redistribute this software for **non-commercial research purposes**, provided appropriate attribution is given. Commercial use requires prior permission from the copyright holder.
