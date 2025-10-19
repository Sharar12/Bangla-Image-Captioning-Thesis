# Thesis Methodology: Bangla Image Captioning

## 1. Problem Formulation
Given an image $I$, the objective is to generate a natural Bengali caption $Y = (y_1, y_2, \dots, y_T)$ maximizing the log-likelihood:
$$\theta^* = \arg\max_\theta \sum_{(I, Y)} \sum_{t=1}^T \log P(y_t \mid y_{<t}, I; \theta)$$

## 2. Model Components
1. **Vision Encoder**:
   - ResNet-50 / ViT-Base / SigLIP2
   - Spatial patch token extraction: $V \in \mathbb{R}^{N \times D_v}$
2. **Attention Mechanism**:
   - Additive Bahdanau Attention computing alignment scores $\alpha_{t, i}$
   - Context vector $c_t = \sum_{i=1}^N \alpha_{t, i} v_i$
3. **Decoders**:
   - Gated Recurrent Unit (GRU) with attention
   - Long Short-Term Memory (LSTM) with dual hidden and cell states
   - BanglaGPT adapter for generative autoregression

## 3. Evaluation Framework
- **BLEU-1 to BLEU-4**
- **ROUGE-L**
- **METEOR**
- **CIDEr**
