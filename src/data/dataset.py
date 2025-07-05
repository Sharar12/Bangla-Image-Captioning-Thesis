import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any

class BanglaImageCaptionDataset(Dataset):
    def __init__(self, root_dir: str, annotations: List[Dict[str, Any]], vocab, transform=None):
        self.root_dir = root_dir
        self.annotations = annotations
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        img_id = item['image_file']
        caption = item['caption']

        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized += [self.vocab.stoi["<EOS>"]]

        return image, torch.tensor(numericalized)
