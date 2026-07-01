# Bangla Image Captioning with Multi-Decoder Fusion Transformer

## Thesis: Encoder-Decoder Architectures for Bengali Image Captioning

A comprehensive research project exploring **Bengali (Bangla) image captioning** using various encoder-decoder architectures. This work benchmarks multiple vision encoders combined with different text decoders to generate fluent Bengali descriptions from images.

---

## Project Overview

This thesis investigates the problem of **automatic image captioning in the Bengali language** — a low-resource language scenario. The project implements and evaluates a range of encoder-decoder architectures, from classical CNN-RNN models to modern transformer-based approaches, using a custom-built 60K hybrid dataset of Bengali-captioned images.

### Best Performing Model

**SigLIP2 (Vision Encoder) + GRU + LSTM + BanglaGPT (Text Decoder)**

```
Image → SigLIP2 → Linear Projection → GRU (4-layer + Attention) → LSTM (4-layer + Attention) → BanglaGPT → Bengali Caption
```

- **BLEU-1**: 0.48 | **BLEU-4**: 0.12 | **METEOR**: 0.29
- Trained on 44K images (BanglaVision40k + V5 Beyond Caption Dataset)
- Vocabulary: 34,365 Bengali words
- Caption length: up to 26 tokens

---

## Repository Structure

```
D:\Python Projects\
│
├── A_Final_Final_Final/              # Primary experiment: SigLIP2 + Att-GRU + LSTM + BanglaGPT
│   ├── EXP_siglip2_Att_GRU_Ex/       # Main experiment workspace
│   │   ├── code/                     # Training, testing, evaluation scripts
│   │   ├── dataset40k/               # 44K consolidated dataset (images + captions)
│   │   ├── output/                   # Trained checkpoints & results
│   │   └── image_results/            # Training curves & metrics charts
│   └── model_result.py               # Results visualization
│
├── Train Full dataset with final model - For Thesis/  # 6-model comparison
│   ├── google-siglip2-base-patch32-256/   # SigLIP2 + Custom Transformer Decoder
│   ├── google-vit-base-patch16-224/       # ViT + Custom Transformer Decoder
│   ├── siglip2_Att_GRU/                   # SigLIP2 + Attentional GRU
│   ├── siglip2_banglabert/                # SigLIP2 + BanglaBERT
│   ├── siglip2_BanglaGPT/                 # SigLIP2 + BanglaGPT
│   ├── siglip2_qwen2/                     # SigLIP2 + Qwen2
│   ├── Final_Hybrid_Dataset_60k/          # 60K hybrid dataset
│   └── image_results/                     # Comparison charts
│
├── Train Full dataset with final model/   # 5 vision encoder comparison
│   ├── google-siglip2-base-patch32-256/   # SigLIP2
│   ├── google-vit-base-patch16-224/       # ViT
│   ├── openai-clip-vit-base-patch16/      # CLIP
│   ├── microsoft-Florence-2-base/         # Florence-2
│   ├── nvidia-C-RADIOv2-B/               # C-RADIOv2
│   ├── Final_Hybrid_Dataset_60k/          # 60K dataset
│   └── image_results/
│
├── Train Text Decoder/                    # Standalone Bangla text decoder (tiny LLaMA)
│   ├── train.py                           # LLaMA training script
│   ├── test.py                            # Caption generation test
│   ├── captions.txt                       # Training captions
│   └── BanglaTDM/                         # Trained Bangla Text Decoder Model
│
├── Jupiter Notebook/                      # Core training & testing scripts
│   ├── claude_train.py                    # Full training pipeline
│   ├── claude_test.py                     # Full evaluation pipeline
│   ├── upload.py                          # Flask web app for captioning
│   └── analyze_test.py                    # Flask web app with metric scoring
│
├── Hybrid_60k_Master_Dataset_Metadata/    # 60K dataset builder
│   ├── local_dataset_builder.py           # Build the 60K hybrid dataset
│   ├── hybrid_60k_master.csv              # Image-caption mapping
│   └── train.txt / val.txt / test.txt     # Data splits
│
├── Research Papers Test/                  # Ablation & historical experiments
│   ├── siglip2_Att_GRU/                   # SigLIP2 + Attentional GRU
│   ├── siglip2_Att_GRU_Ex/               # SigLIP2 + GRU Extended
│   ├── siglip2_banglabert/                # SigLIP2 + BanglaBERT
│   ├── siglip2_banglagpt/                 # SigLIP2 + BanglaGPT
│   ├── Siglip2_cusDecoder/                # SigLIP2 + Custom Decoder
│   ├── swin_banglabert/                   # Swin Transformer + BanglaBERT
│   ├── vit_banglabert/                    # ViT + BanglaBERT
│   ├── inception_gru/                     # Inception + GRU
│   └── xception_bigru_attention/          # Xception + BiGRU + Attention
│
├── Test/                                  # GPU stress test & environment test
│   └── requirements_stable.txt            # Pinned dependency versions
│
└── batch_evaluation_results.csv           # Summary comparison across 5 encoders
```

---

## Models & Architectures

### Vision Encoders Evaluated

| Encoder | Params | BLEU-1 | BLEU-4 | METEOR |
|---------|--------|--------|--------|--------|
| **SigLIP2** (google/siglip2-base-patch16-224) | 0.4B | **0.4800** | **0.1206** | **0.2914** |
| **SigLIP2** (google/siglip2-base-patch32-256) | 0.4B | — | — | — |
| **CLIP** (openai/clip-vit-base-patch32) | 0.4B | 0.3744 | 0.0770 | 0.2159 |
| **DINOv2** (facebook/dinov3-vits16) | — | 0.2789 | 0.0508 | 0.2100 |
| **C-RADIOv2** (nvidia/C-RADIOv2-B) | 98M | 0.2352 | 0.0226 | 0.1552 |
| **U-DOP** (microsoft/udop-large) | — | 0.0780 | 0.0036 | 0.0925 |

### Text Decoders Evaluated

- **Attentional GRU** (4-layer, Bahdanau attention)
- **Attentional LSTM** (4-layer, Bahdanau attention)
- **BanglaGPT** (shahidul034/BanglaGPT — GPT-style Bengali LM)
- **BanglaBERT** (BERT-style Bengali language model)
- **BanglaTDM** — Custom tiny LLaMA trained from scratch
- **Custom Transformer Decoder**
- **Qwen2** (Qwen2 language model)

### Final Architecture (Multi-Decoder Fusion)

```
Encoder: SigLIP2 (google/siglip2-base-patch32-256)
  → Unfreeze last 4 layers for fine-tuning
  → Output: 768-dim visual features

Projection: Linear(768 → 512)

Decoder Pipeline:
  1. GRUDecoder (4-layer GRU + Bahdanau Attention)
     → Processes embedded tokens with visual context
  2. LSTMDecoder (4-layer LSTM + Bahdanau Attention)
     → Refines GRU output with visual context
  3. BanglaGPTDecoder (shahidul034/BanglaGPT)
     → Unfreeze last 4/12 layers
     → Projects fused features → generates token logits

Output: CrossEntropyLoss with label smoothing (0.1)
```

---

## Datasets

### Primary: Final Hybrid Dataset (60K images)

| Source | Images | Language | Description |
|--------|--------|----------|-------------|
| **Flickr30k** | ~30K | Bengali (translated) | Flickr30k with Bengali captions |
| **BNature** | ~15K | Bengali | Nature images with Bengali captions |
| **COCO 2017** | ~15K | Bengali (translated) | Common Objects in Context |

### Secondary: BanglaVision40k (44K images)

| Source | Images | Description |
|--------|--------|-------------|
| **BanglaView** | ~20K | Bangla web images |
| **BNature** | ~10K | Nature images |
| **Bornon** | ~10K | Bangla cultural images |
| **V5 Beyond Caption** | ~4K | Additional captioned images |

### Dataset Building

The `Hybrid_60k_Master_Dataset_Metadata/local_dataset_builder.py` script:
- Takes a CSV mapping (Flickr30k + BNature + COCO)
- Copies and renames images into a unified folder structure
- Supports auto-download of COCO images from the web
- Outputs to `Final_Hybrid_Dataset_60k/`

---

## Training Configurations

### Competition Mode (Encoder Comparison)
```
BATCH_SIZE = 64
EPOCHS = 10
MAX_LR = 1.5e-4
NUM_LAYERS = 6
HIDDEN_DIM = 2048
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
```

### High IQ Mode (Best Performance)
```
BATCH_SIZE = 32
EPOCHS = 20
MAX_LR = 1e-4
NUM_LAYERS = 16
HIDDEN_DIM = 4096
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
```

### Grand Finale (100 Epochs)
```
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS = 100
MAX_LR = 1e-4
NUM_LAYERS = 16
HIDDEN_DIM = 4096
DROPOUT = 0.2
WEIGHT_DECAY = 0.05
```

---

## Web Applications

Two Flask-based web demos are included:

- **`upload.py`** (port 8182): Simple image upload → caption generation
- **`analyze_test.py`** (port 8183): Upload + metric scoring against reference captions

Both feature dark-mode UI with drag-and-drop image upload.

---

## Requirements

- Python 3.11+
- PyTorch 2.11+
- CUDA-capable GPU (12GB+ VRAM recommended)
- See `Test/requirements_stable.txt` for full pinned dependencies

Key dependencies: `torch`, `torchvision`, `transformers`, `pillow`, `tqdm`, `pandas`, `matplotlib`, `flask`, `nltk` (for BLEU/METEOR/ROUGE-L/CIDEr scoring)

---

## Results & Sample Captions

The model generates Bengali captions such as:

| Image Content | Generated Caption |
|--------------|-------------------|
| Child walking on path | একটি সাদা পোশাক পরা একটি ছোট মেয়ে একটি পথে হাঁটছে |
| People at beach | সমুদ্র সৈকতে দাঁড়িয়ে থাকা বেশ কয়েকজন লোক |
| Food on table | টেবিলের উপর রাখা বিভিন্ন ধরনের খাবার |

---

## Key Findings

1. **SigLIP2** consistently outperforms other vision encoders (CLIP, ViT, Florence-2, C-RADIOv2) for Bengali captioning
2. **Multi-decoder fusion** (GRU → LSTM → BanglaGPT) outperforms single-decoder architectures
3. **Unfreezing the last 4 layers** of both encoder and decoder provides the best fine-tuning balance
4. The 60K hybrid dataset significantly improves generalization over the 40K dataset
5. BanglaGPT as a frozen language model with fine-tuned top layers is effective for low-resource Bengali generation

---

## Citation

If you use this work in your research, please cite:

```
@thesis{sharar2025bangla,
  title={Bengali Image Captioning using Encoder-Decoder Architectures},
  author={Sharar Hossain},
  year={2025},
  school={[Your University]}
}
```

---

## License

This project is for academic research purposes.

## Author

**Sharar Hossain** — [GitHub](https://github.com/Sharar12)
