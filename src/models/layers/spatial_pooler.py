import torch
import torch.nn as nn

class SpatialPooler(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(embed_dim, 1)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, D)
        weights = torch.softmax(self.attn_weights(visual_features), dim=1)
        pooled = (visual_features * weights).sum(dim=1)
        return pooled
