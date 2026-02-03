# Focal Loss vs Label Smoothing Exploration

Ablation study on rare inflectional suffixes:
- Standard Cross-Entropy: BLEU-4 = 31.60 | Rare Token F1 = 28.4%
- Label Smoothing (0.1): BLEU-4 = 32.40 | Rare Token F1 = 31.2%
- Focal Loss (gamma=2.0): **BLEU-4 = 33.10** | **Rare Token F1 = 35.8%**
