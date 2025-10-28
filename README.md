# Bangla Image Captioning Thesis

A thesis/research repository for **Bangla image captioning**, focused on generating Bengali descriptions from images using deep-learning-based vision and language models.

[![Python CI](https://github.com/Sharar12/Bangla-Image-Captioning-Thesis/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Sharar12/Bangla-Image-Captioning-Thesis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

The project explores image-to-text generation for Bangla, including dataset preparation, model training, evaluation, and experiment tracking.

## Research Focus

- Bangla image caption generation
- Vision-language modeling
- Dataset preparation and preprocessing
- Sequence generation and decoding
- Automatic evaluation with captioning metrics (BLEU, ROUGE-L, METEOR, CIDEr)
- Reproducible training experiments

## Architecture

The **BanglaVision** framework incorporates:
1. **Vision Encoders**: ResNet-50, ViT-Base, CLIP, and pretrained **SigLIP2**.
2. **Attention Mechanisms**: Additive Bahdanau Attention and Adaptive Visual Sentinel gates.
3. **Decoders**: Bahdanau GRU/LSTM decoders and causal BanglaGPT adapter layers.
4. **Generation**: Beam Search with length penalty and stochastic Top-K / Nucleus sampling.

```text
Images + Bangla Captions
          ↓
   Dataset Preparation
          ↓
  Vision Encoder (SigLIP2 / ViT / ResNet)
          ↓
  Bahdanau Attention / BanglaGPT Adapter
          ↓
      Decoder Model
          ↓
  Bangla Caption Generation (Beam Search)
          ↓
   Evaluation (BLEU, ROUGE, METEOR, CIDEr)
```

## Benchmark Results

Evaluated on the **BanglaLekha-ImageCaption** benchmark:

| Architecture | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 + GRU | 48.20 | 18.50 | 42.10 | 21.30 | 0.65 |
| ResNet-50 + LSTM | 49.70 | 19.80 | 43.50 | 22.10 | 0.68 |
| ViT-Base + GRU | 53.40 | 23.10 | 46.80 | 24.50 | 0.79 |
| CLIP ViT-B/16 + LSTM | 56.10 | 25.40 | 49.20 | 26.20 | 0.88 |
| **SigLIP2 + BanglaGPT (Proposed)** | **62.80** | **31.60** | **55.40** | **31.00** | **1.14** |

## Quick Start

### 1. Installation
```bash
git clone https://github.com/Sharar12/Bangla-Image-Captioning-Thesis.git
cd Bangla-Image-Captioning-Thesis
pip install -r requirements.txt
```

### 2. Training
```bash
python train.py --config configs/siglip2_banglagpt.yaml
```

### 3. Interactive Web Demo
```bash
python app.py
```

## Reproducibility

Large datasets and model checkpoints are stored externally due to GitHub repository size limits. Refer to [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for data download and setup instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
