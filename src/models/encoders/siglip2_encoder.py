import torch
import torch.nn as nn
from transformers import AutoModel
from src.models.encoders.base_encoder import BaseVisionEncoder

class SigLIP2VisionEncoder(BaseVisionEncoder):
    """SigLIP2 provides rich multilingual-aligned visual representations."""
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", embed_dim: int = 512):
        super().__init__(embed_dim)
        model = AutoModel.from_pretrained(model_name)
        self.vision_model = model.vision_model
        hidden_dim = self.vision_model.config.hidden_size
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(images)
        # Extract spatial token representations
        patch_tokens = outputs.last_hidden_state
        return self.proj(patch_tokens)
