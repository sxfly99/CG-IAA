# CG-IAA: Towards Explainable Image Aesthetics Assessment with Attribute-Oriented Critiques Generation

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TCSVT-blue)](https://ieeexplore.ieee.org/document/10700814)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **"Towards Explainable Image Aesthetics Assessment with Attribute-oriented Critiques Generation"** (IEEE TCSVT 2025).

---

## 📰 News

-  🎉 We release the **multi-attribute aesthetic critiques generation model** with pre-trained weights and training data!
-  🎉 Our CG-IAA paper was accepted by IEEE TCSVT!
- **[Coming Soon]** The complete aesthetic assessment model will be released soon.

---

## 💡 Overview

**CG-IAA** addresses a critical challenge in image aesthetics assessment: How can we leverage the power of multimodal learning when aesthetic critiques are unavailable? Our solution generates high-quality aesthetic critiques from multiple attribute perspectives, enabling both accurate aesthetic prediction and enhanced model explainability.

### Key Contributions

- **Multi-Attribute Aesthetic Critiques Generation**: We propose a CLIP-based model that generates diverse aesthetic critiques from four different perspectives:
  - 🎨 **Color and Light**: Color harmony, saturation, lighting quality
  - 📐 **Composition**: Layout, balance, structural elements
  - 🔍 **Depth and Focus**: Depth of field, focus, blur effects
  - ⭐ **General Feelings**: Overall aesthetic impression and quality

- **Enhanced Explainability**: Generated critiques provide human-readable explanations for aesthetic judgments, making the model more transparent and interpretable.

### Framework Architecture

<p align="center">
    <img src="assets/pipeline.png" alt="CG-IAA Pipeline" width="800" />
</p>

The CG-IAA framework consists of three main components:
1. **VLAP (Vision-Language Aesthetic Pretraining)**: Fine-tune CLIP on aesthetic data
2. **MAEL (Multi-Attribute Experts Learning)**: Train attribute-specific expert models
3. **MAP (Multimodal Aesthetics Prediction)**: Fuse visual and textual features for final prediction

---

## 🚀 What's Released

### ✅ Currently Available

1. **Aesthetic Critiques Generation Model** - Multi-attribute aesthetic critiques generation
   - Pre-trained model weights
   - Inference code for single image processing

2. **Training Data** - Large-scale multi-attribute aesthetic critique dataset
   - ~150K critiques for Color and Light
   - ~100K critiques for Composition
   - ~120K critiques for Depth and Focus
   - ~570K critiques for General Feelings
   - Total: **~940K aesthetic critiques** with attribute annotations

### 🔜 Coming Soon

- Complete aesthetic assessment model

---

## 📦 Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/your-username/CG-IAA.git
cd CG-IAA

# Create and activate conda environment
conda env create -f environment.yml
conda activate cg-iaa
```

### Download Pre-trained Weights

Download the pre-trained model weights from Google Drive and place them in the `checkpoints/` directory:

📥 **[Download Model Weights](https://drive.google.com/drive/folders/12jO6mF3ppBpap1tOx3ic2Mo4YNZ7m14Y?usp=drive_link)** 
The checkpoints directory should contain:
```
checkpoints/
├── base_model.pt          # Base model
├── color.pt              # Color expert model
├── composition.pt        # Composition expert model
├── dof.pt               # Depth of Field expert model
└── general.pt           # General expert model
```

### Download Data (Optional)

If you want to train your own models, download our multi-attribute aesthetic critique dataset:

📥 **[Download Training Data](https://drive.google.com/drive/folders/1cKLD2pl405Wl2UB2RXMKilzOLS4IJpeH?usp=drive_link)**

---

## 🎯 Quick Start

### Single Image Inference

Generate aesthetic critiques for a single image:

```bash
python caption_inference.py --image_path samples/1.jpg
```

**Output:**
```
================================================================================
Multi-Attribute Aesthetic Captions for: samples/1.jpg
================================================================================

[Color]

[Composition]

[Depth of Field]

[General]

================================================================================
```

---

## 📊 Model Performance

Our generated aesthetic critiques achieve competitive performance when used alone for IAA task:

| Method | PLCC ↑ | SRCC ↑ | ACC ↑ |
|--------|--------|--------|-------|
| ARIC (AAAI 2023) | 0.591 | 0.550 | 74.3 |
| VILA (CVPR 2023) | 0.534 | 0.505 | 75.2 |
| **AesCritique (Ours)** | **0.720** | **0.712** | **80.8** |

*Tested on AVA database using text-only input*

---

## 📁 Dataset Structure

Our released multi-attribute aesthetic critique dataset is organized as follows:

```
data/
├── color.json           # Color and Light critiques
├── composition.json     # Composition critiques
├── dof.json            # Depth and Focus critiques
└── general.json        # General Feelings critiques
```

Each JSON file contains entries in the following format:
```json
[
  {
    "id": 0,
    "img_id": "773931",
    "caption": "Image feels a tad dark, which I dont think helps this image for me."
  },
  ...
]
```

---

## 🙏 Acknowledgement

CG-IAA is built upon the following excellent open-source projects:

- [CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training
- [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) - CLIP Prefix for Image Captioning
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models

---

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{li2025cgiaa,
  author={Li, Leida and Sheng, Xiangfei and Chen, Pengfei and Wu, Jinjian and Dong, Weisheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Towards Explainable Image Aesthetics Assessment With Attribute-Oriented Critiques Generation}, 
  year={2025},
  volume={35},
  number={2},
  pages={1464-1477}
}
```
