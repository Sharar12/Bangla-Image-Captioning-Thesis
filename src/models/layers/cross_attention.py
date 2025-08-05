import torch
import torch.nn as nn

class VisualCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, visual_context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(query, visual_context, visual_context)
        return self.norm(query + attn_out)
