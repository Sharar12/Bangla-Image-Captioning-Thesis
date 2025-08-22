import torch
import torch.nn as nn

class VisualSentinelGate(nn.Module):
    """Allows the decoder to decide how much to rely on image vs previous text context."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, context_vector: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        beta = self.gate(torch.cat([context_vector, hidden_state], dim=-1))
        return beta * context_vector + (1.0 - beta) * hidden_state
