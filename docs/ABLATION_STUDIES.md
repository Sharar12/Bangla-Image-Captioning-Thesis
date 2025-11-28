# Ablation Studies & Hyperparameter Optimization

## Encoder Dimension vs Decoding Quality
- ResNet-50 (2048): Fast convergence, lower semantic fidelity.
- ViT-Base (768): Strong patch granularity, higher computational cost.
- SigLIP2 (768): Superior multi-modal alignment on Bengali descriptions (+3.2 BLEU-4).

## Decoder Hidden States
- 256 dimensions: Underspecified for complex compound Bengali grammar.
- 512 dimensions: Optimal balance between parameter budget and CIDEr score (1.14).
