import torch
import torch.nn as nn
from src.models.encoders.base_encoder import BaseVisionEncoder

class Florence2Backbone(BaseVisionEncoder):
    def __init__(self, embed_dim: int = 512):
        super().__init__(embed_dim)
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Stub for Florence-2 feature projection
        b = images.shape[0]
        dummy_patches = torch.zeros(b, 196, 768, device=images.device)
        return self.proj(dummy_patches)
