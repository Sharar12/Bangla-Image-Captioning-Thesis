import torch
import torch.nn as nn

class PrefixTuningLayer(nn.Module):
    def __init__(self, prefix_len: int = 10, embed_dim: int = 768):
        super().__init__()
        self.prefix_len = prefix_len
        self.prefix_embedding = nn.Parameter(torch.randn(1, prefix_len, embed_dim) * 0.01)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.prefix_embedding.expand(batch_size, -1, -1)
