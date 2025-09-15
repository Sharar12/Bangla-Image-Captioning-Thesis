import torch
import torch.nn as nn

class BanglaGPTAdapter(nn.Module):
    """Adapter projecting vision encoder tokens into BanglaGPT token embedding space."""
    def __init__(self, visual_dim: int, gpt_embed_dim: int = 768, num_visual_tokens: int = 16):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, gpt_embed_dim),
            nn.GELU(),
            nn.Linear(gpt_embed_dim, gpt_embed_dim),
            nn.LayerNorm(gpt_embed_dim)
        )
        self.token_resampler = nn.AdaptiveAvgPool1d(num_visual_tokens)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: (B, N, visual_dim)
        proj_feats = self.proj(visual_features) # (B, N, gpt_embed_dim)
        # Resample to fixed prefix length: (B, gpt_embed_dim, N) -> (B, gpt_embed_dim, num_visual_tokens)
        resampled = self.token_resampler(proj_feats.transpose(1, 2)).transpose(1, 2)
        return resampled
