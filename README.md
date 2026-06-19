# Bangla Image Captioning Thesis (BanglaVision v2.0.0)

[![Python CI](https://github.com/Sharar12/Bangla-Image-Captioning-Thesis/actions/workflows/python-tests.yml/badge.svg)](https://github.com/Sharar12/Bangla-Image-Captioning-Thesis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version: 2.0.0](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](https://github.com/Sharar12/Bangla-Image-Captioning-Thesis/releases)

A research thesis framework for **Bangla image captioning**, generating natural Bengali descriptions from images using deep vision-language architectures.

---

## Highlights in v2.0.0
- **SigLIP2 + Cross-Attention Gating**: State-of-the-art vision-language multi-modal alignment.
- **Self-Critical Sequence Training (SCST)**: Reinforcement learning directly optimizing CIDEr score (+0.18 CIDEr gain).
- **Diverse Beam Search**: Exploration of candidate captions with Hamming distance diversity penalty.
- **INT8 Quantization & ONNX Engine**: Ultra-fast CPU inference at 48 ms per image.
- **Interactive Web Demo & FastAPI**: Full interactive Gradio suite and production REST API.

---

## Final Benchmark Comparison

| Architecture | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 + GRU Baseline | 48.20 | 18.50 | 42.10 | 21.30 | 0.65 |
| ViT-Base + GRU | 53.40 | 23.10 | 46.80 | 24.50 | 0.79 |
| SigLIP2 + BanglaGPT (CE) | 62.80 | 31.60 | 55.40 | 31.00 | 1.14 |
| **SigLIP2 + SCST (Proposed, v2.0)** | **65.40** | **34.20** | **58.10** | **33.50** | **1.32** |

---

## Quick Start

```bash
git clone https://github.com/Sharar12/Bangla-Image-Captioning-Thesis.git
cd Bangla-Image-Captioning-Thesis
pip install -r requirements.txt

# Run Gradio Web Demo
python app.py

# Run FastAPI Server
uvicorn src.api.server:app --port 8000
```

## Citation
Please refer to `CITATION.cff` for thesis citation format.

## License
MIT License.
