# Bangla Image Captioning Thesis

A thesis/research repository for **Bangla image captioning**, focused on generating Bengali descriptions from images using deep-learning-based vision and language models.

## Overview

The project explores image-to-text generation for Bangla, including dataset preparation, model training, evaluation, and experiment tracking.

## Research Focus

- Bangla image caption generation
- Vision-language modeling
- Dataset preparation and preprocessing
- Sequence generation and decoding
- Automatic evaluation with captioning metrics
- Reproducible training experiments

## Related Architecture

The later BanglaVision implementation in this project family uses a pretrained **SigLIP2** vision encoder with GRU/LSTM/BanglaGPT decoder components. The exact experiment configuration should be taken from the corresponding training scripts and result files in this repository.

## Typical Workflow

```text
Images + Bangla Captions
          ↓
   Dataset Preparation
          ↓
   Vision Encoder
          ↓
     Decoder Model
          ↓
  Bangla Caption Generation
          ↓
        Evaluation
```

## Evaluation

Typical image-captioning evaluation includes BLEU, METEOR, ROUGE-L, and CIDEr.

## Reproducibility

Large datasets and model checkpoints may not be stored in GitHub because of repository/file-size limits. Use the dataset/model instructions included in the repository when reproducing experiments.

## Purpose

Academic thesis and research work in Bangla vision-language generation.

## License

For academic and research purposes.