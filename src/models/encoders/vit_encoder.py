import torch
import torch.nn as nn
from transformers import ViTModel
from src.models.encoders.base_encoder import BaseVisionEncoder

class ViTEncoder(BaseVisionEncoder):
    def __init__(self, model_name: str = "google/vit-base-patch16-224", embed_dim: int = 512):
        super().__init__(embed_dim)
        self.vit = ViTModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.vit.config.hidden_size, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(images)
        # Exclude [CLS] token, keep patch representations: (B, 196, hidden)
        patch_feats = outputs.last_hidden_state[:, 1:, :]
        return self.proj(patch_feats)
