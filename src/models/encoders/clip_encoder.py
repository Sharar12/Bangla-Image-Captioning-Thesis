import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from src.models.encoders.base_encoder import BaseVisionEncoder

class ClipVisionEncoder(BaseVisionEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", embed_dim: int = 512):
        super().__init__(embed_dim)
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.clip.config.hidden_size, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.clip(images)
        patch_feats = outputs.last_hidden_state[:, 1:, :]
        return self.proj(patch_feats)
