import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseVisionEncoder(nn.Module, ABC):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual spatial features: (B, num_patches, embed_dim)"""
        pass
