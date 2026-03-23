import torch
import torch.nn as nn
from src.models.encoders.siglip_encoder import SigLIP2VisionEncoder
from src.models.decoders.bangla_gpt_adapter import BanglaGPTAdapter

class BanglaVisionCaptioner(nn.Module):
    def __init__(self, freeze_encoder: bool = True, gpt_dim: int = 768):
        super().__init__()
        self.encoder = SigLIP2VisionEncoder(freeze=freeze_encoder)
        self.adapter = BanglaGPTAdapter(visual_dim=self.encoder.embed_dim, gpt_embed_dim=gpt_dim)
        self.gpt_dim = gpt_dim
        self._cached_features = None

    def encode_image(self, images: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        if use_cache and self._cached_features is not None:
            return self._cached_features
        visual_tokens = self.encoder(images)
        prefix_embeddings = self.adapter(visual_tokens)
        if use_cache:
            self._cached_features = prefix_embeddings
        return prefix_embeddings
