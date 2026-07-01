"""
BanglaCaptionDataset and Vocabulary — extracted to a module so DataLoader
multiprocessing works in Jupyter notebooks.
"""
import os
import json
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor
from PIL import Image

# Default max sequence length (overridden at dataset creation)
DEFAULT_MAX_SEQ_LEN = 26


class Vocabulary:
    """Bidirectional word-to-index and index-to-word mapping."""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.count = 4

    def add_word(self, word):
        """Add a word to the vocabulary if not already present."""
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1

    def build_from_captions(self, captions_dict, min_freq=1):
        """Build vocabulary from a dictionary of captions, filtering by frequency."""
        freq = Counter()
        for caps in captions_dict.values():
            for cap in caps:
                for word in cap.split():
                    freq[word] += 1
        for word, count in freq.items():
            if count >= min_freq:
                self.add_word(word)

    def encode(self, caption, max_len):
        """Encode a caption string into token IDs with padding."""
        tokens = caption.split()
        ids = (
            [self.word2idx["<start>"]]
            + [self.word2idx.get(w, self.word2idx["<unk>"]) for w in tokens]
            + [self.word2idx["<end>"]]
        )
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.word2idx["<end>"]]
        ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids[:max_len], dtype=torch.long)

    def decode(self, ids):
        """Decode token IDs back into a caption string."""
        words = []
        for idx in ids:
            w = self.idx2word.get(idx, "<unk>")
            if w == "<end>":
                break
            if w not in ("<pad>", "<start>"):
                words.append(w)
        return " ".join(words)

    @property
    def vocab_size(self):
        return self.count

    def save(self, path):
        """Save vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "idx2word": {str(k): v for k, v in self.idx2word.items()},
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """Load vocabulary from a JSON file."""
        v = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v.word2idx = data["word2idx"]
        v.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        v.count = len(v.word2idx)
        return v


class BanglaCaptionDataset(Dataset):
    """PyTorch Dataset for Bangla image captioning."""

    def __init__(
        self, image_list, captions, vocab, image_dir, processor_path,
        is_train=True, max_len=None
    ):
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_len = max_len if max_len is not None else DEFAULT_MAX_SEQ_LEN
        self.is_train = is_train
        self.processor = AutoImageProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        self.aug = (
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1
                ),
            ])
            if is_train
            else None
        )
        self.pairs = []
        for img_name in image_list:
            caps = captions.get(img_name, [""])
            for cap in caps:
                self.pairs.append((img_name, cap))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, cap = self.pairs[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
            if self.aug:
                image = self.aug(image)
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 256, 256)
        caption_ids = self.vocab.encode(cap, self.max_len)
        return pixel_values, caption_ids
