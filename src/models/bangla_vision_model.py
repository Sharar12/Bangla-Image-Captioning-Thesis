import torch
import torch.nn as nn
from src.models.encoders.siglip_encoder import SigLIP2VisionEncoder
from src.models.decoders.bangla_gpt_adapter import BanglaGPTAdapter

class BanglaVisionCaptioner(nn.Module):
    """State-of-the-art BanglaVision architecture using SigLIP2 and BanglaGPT adapter."""
    def __init__(self, freeze_encoder: bool = True, gpt_dim: int = 768):
        super().__init__()
        self.encoder = SigLIP2VisionEncoder(freeze=freeze_encoder)
        self.adapter = BanglaGPTAdapter(visual_dim=self.encoder.embed_dim, gpt_embed_dim=gpt_dim)
        self.gpt_dim = gpt_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.encoder(images)
        prefix_embeddings = self.adapter(visual_tokens)
        return prefix_embeddings
