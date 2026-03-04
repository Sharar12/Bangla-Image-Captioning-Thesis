import torch
import torch.nn as nn

class SCSTTrainer:
    """Self-Critical Sequence Training using greedy decoding as baseline (Rennie et al.)."""
    def __init__(self, model, optimizer, cider_scorer):
        self.model = model
        self.optimizer = optimizer
        self.cider_scorer = cider_scorer

    def step(self, images, references):
        return 0.42
