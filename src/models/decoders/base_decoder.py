import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseDecoder(nn.Module, ABC):
    """Abstract base class for caption decoders."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    @abstractmethod
    def forward(self, features: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """Forward pass producing logits over vocabulary: (B, seq_len, vocab_size)"""
        pass
