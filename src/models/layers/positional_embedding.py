import torch
import torch.nn as nn

class SpatialPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, grid_size: int = 14):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, embed_dim) * 0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return patch_tokens + self.pos_embed
