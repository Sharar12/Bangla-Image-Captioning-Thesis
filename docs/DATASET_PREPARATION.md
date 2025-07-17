# 60K Hybrid Bengali Image-Caption Dataset Preparation

## 📊 Dataset Composition
1. **BanglaLekhaImageCaption**: 9,150 images with crowd-sourced authentic Bangla captions.
2. **Bornona Dataset**: 15,000 images with descriptive regional captions.
3. **COCO-Bangla (Translated & Quality Screened)**: 36,000 diverse daily scenes.
**Total**: ~60,150 image-caption pairs.

## 🧹 Preprocessing Rules
- Unicode NFC normalization for all Bengali graphemes.
- Filtering of duplicate captions and whitespace irregularities.
- Vocabulary pruning at minimum occurrence threshold of 3.
