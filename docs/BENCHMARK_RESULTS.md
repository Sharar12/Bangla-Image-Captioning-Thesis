# Experimental Benchmark Results

Evaluation conducted on **BanglaLekha-ImageCaption** and **Chittron** Bengali captioning test sets.

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-50 + Bahdanau GRU | 48.20 | 33.10 | 24.60 | 18.50 | 42.10 | 21.30 | 0.65 |
| ResNet-50 + Bahdanau LSTM | 49.70 | 34.80 | 25.90 | 19.80 | 43.50 | 22.10 | 0.68 |
| ViT-Base + Bahdanau GRU | 53.40 | 38.60 | 29.40 | 23.10 | 46.80 | 24.50 | 0.79 |
| CLIP ViT-B/16 + LSTM | 56.10 | 41.50 | 32.20 | 25.40 | 49.20 | 26.20 | 0.88 |
| **SigLIP2 + BanglaGPT (Proposed)** | **62.80** | **48.90** | **39.50** | **31.60** | **55.40** | **31.00** | **1.14** |

## Key Insights
- Vision Transformer backbones consistently outperform convolutional feature extractors in spatial context alignment.
- SigLIP2 vision encoder pretrained representations transfer exceptionally well to Bengali linguistic contexts.
- Beam search with length penalty $lpha=0.7$ achieved the highest CIDEr consensus ratings.
