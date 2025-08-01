import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.models.encoders.base_encoder import BaseVisionEncoder

class ResNetEncoder(BaseVisionEncoder):
    def __init__(self, embed_dim: int = 512):
        super().__init__(embed_dim)
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove fc and avgpool
        self.conv_layers = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 2048, 7, 7) -> (B, 49, 2048) -> (B, 49, embed_dim)
        feat = self.conv_layers(images)
        b, c, h, w = feat.shape
        feat = feat.view(b, c, -1).permute(0, 2, 1)
        return self.proj(feat)
