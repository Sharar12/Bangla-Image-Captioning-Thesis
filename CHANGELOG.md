# Changelog

All notable changes to the Bangla Image Captioning Thesis project are documented in this file.

## [1.0.0] - 2025-10-28
### Added
- Complete data preprocessing pipeline for Bengali text normalization and tokenization.
- Vision encoder implementations for ResNet-50, ViT-Base, CLIP, and SigLIP2.
- Bahdanau additive attention layer and adaptive visual sentinel gating.
- GRU and LSTM attention decoders with scheduled teacher forcing.
- BanglaGPT causal projection adapter for vision-language generation.
- Beam search decoder with length penalty and top-k/top-p sampling.
- Full evaluation suite covering BLEU-1..4, ROUGE-L, METEOR, and CIDEr.
- Interactive Gradio web demonstration interface (`app.py`).
- Docker containerization and GitHub Actions continuous integration.
- Comprehensive thesis experimental benchmark documentation.
