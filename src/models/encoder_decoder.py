import torch
import torch.nn as nn
from src.models.encoders.base_encoder import BaseVisionEncoder
from src.models.decoders.base_decoder import BaseDecoder

class BanglaCaptioningModel(nn.Module):
    """Unified End-to-End Image Captioning Architecture."""
    def __init__(self, encoder: BaseVisionEncoder, decoder: BaseDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, teacher_forcing_ratio=teacher_forcing_ratio)
        return outputs
