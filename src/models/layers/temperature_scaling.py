import torch
import torch.nn as nn

class CrossAttentionTemperatureScaler(nn.Module):
    """Dynamic temperature scaler controlling cross-attention sharpness during decoding."""
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        clamped_temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        return attention_scores / clamped_temp
