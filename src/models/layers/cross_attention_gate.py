import torch
import torch.nn as nn

class CrossAttentionGate(nn.Module):
    """Sigmoid gating mechanism balancing visual context against textual recurrence."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, visual_ctx: torch.Tensor, text_hidden: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([visual_ctx, text_hidden], dim=-1))
        blended = g * visual_ctx + (1.0 - g) * text_hidden
        return blended
