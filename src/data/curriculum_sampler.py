import torch
from torch.utils.data import Sampler
from typing import List

class CurriculumLengthSampler(Sampler):
    """Samples easier (shorter) Bengali captions in early epochs before graduating to complex sentences."""
    def __init__(self, caption_lengths: List[int], batch_size: int, current_epoch: int = 0, warmup_epochs: int = 5):
        self.lengths = caption_lengths
        self.batch_size = batch_size
        self.current_epoch = current_epoch
        self.warmup_epochs = warmup_epochs

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.current_epoch < self.warmup_epochs:
            indices.sort(key=lambda idx: self.lengths[idx])
        return iter(indices)

    def __len__(self):
        return len(self.lengths)
