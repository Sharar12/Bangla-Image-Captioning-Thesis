"""
fix_train.py — Multi-caption training (5 caps per image per epoch).

HOW TO USE
----------
Replace your project's BanglaCaptionDataset with MultiCaptionDataset below.
The dataset yields 5 (image, caption) pairs per image per epoch by cycling
through all available captions for that image.

For Group A (siglip2_Att_GRU, siglip2_Att_GRU_Ex, EXP_siglip2_Att_GRU_Ex, Siglip2_cusDecoder):
    → Use MultiCaptionDataset (custom vocab variant)

For Group B (xception_bigru_attention, inception_gru):
    → Use MultiCaptionDataset (custom vocab variant)

For Group C (siglip2_banglabert, siglip2_banglagpt, vit_banglabert, swin_banglabert):
    → Use MultiCaptionDataset (HF tokenizer variant)

Instructions in each project's train.py:
  1. Comment out the old BanglaCaptionDataset class
  2. Import or paste MultiCaptionDataset
  3. Instantiate with num_captions=5
"""

import random
import torch
from torch.utils.data import Dataset


class MultiCaptionDataset(Dataset):
    """
    Generic multi-caption dataset that yields NUM_CAPTIONS pairs per image.
    Handles images with fewer captions by cycling (i % num_available).
    """

    def __init__(
        self,
        image_list,
        captions,
        image_dir,
        num_captions=5,
        image_transform=None,
        processor=None,
        vocab=None,
        tokenizer=None,
        max_len=51,
    ):
        self.image_list = image_list
        self.captions = captions
        self.image_dir = image_dir
        self.num_captions = num_captions
        self.transform = image_transform
        self.processor = processor
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pairs = []
        for img_name in image_list:
            caps = captions.get(img_name, [""])
            for i in range(num_captions):
                self.pairs.append((img_name, i % len(caps)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, cap_idx = self.pairs[idx]
        try:
            from PIL import Image

            image = (
                Image.open(self.image_dir / img_name).convert("RGB")
                if isinstance(self.image_dir, (str, bytes))
                else Image.open(self.image_dir / img_name).convert("RGB")
            )
        except Exception:
            image = None

        # --- Image processing ---
        if self.processor is not None:
            # HuggingFace processor (SigLIP2)
            pixel_values = (
                self.processor(images=image, return_tensors="pt")[
                    "pixel_values"
                ].squeeze(0)
                if image is not None
                else torch.zeros(3, 256, 256)
            )
        elif self.transform is not None:
            pixel_values = (
                self.transform(image) if image is not None else torch.zeros(3, 224, 224)
            )
        else:
            pixel_values = torch.zeros(3, 224, 224)

        # --- Caption encoding ---
        cap = self.captions[img_name][cap_idx]

        if self.tokenizer is not None:
            # HF tokenizer path
            tokens = self.tokenizer(
                cap,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return (
                pixel_values,
                tokens["input_ids"].squeeze(0),
                tokens["attention_mask"].squeeze(0),
            )
        elif self.vocab is not None:
            # Custom vocab path
            caption_ids = self.vocab.encode(cap, self.max_len)
            return pixel_values, caption_ids
        else:
            # Raw tensor path
            return pixel_values, torch.tensor([0])


# ─── Example integration snippets ───────────────────────────────────────────

# GROUP A & B — Custom vocab projects (siglip2_Att_GRU, Siglip2_cusDecoder,
#               xception_bigru_attention, inception_gru, etc.)
"""
train_dataset = MultiCaptionDataset(
    train_list, captions, os.path.join(DATA_DIR, "images"),
    num_captions=5,
    processor=train_dataset.processor,       # or image_transform=...
    vocab=vocab,
    max_len=MAX_SEQ_LEN,
)
"""

# GROUP C — HF tokenizer projects (siglip2_banglabert, siglip2_banglagpt,
#            vit_banglabert, swin_banglabert)
"""
train_dataset = MultiCaptionDataset(
    train_list, captions, os.path.join(DATA_DIR, "images"),
    num_captions=5,
    transform=train_dataset.transform,
    tokenizer=tokenizer,
    max_len=MAX_SEQ_LEN,
)
"""
