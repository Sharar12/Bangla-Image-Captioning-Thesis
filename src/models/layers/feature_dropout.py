import torch
import torch.nn as nn

class SpatialPatchDropout(nn.Module):
    """Randomly masks entire spatial visual patches during training to discourage co-adaptation."""
    def __init__(self, drop_rate: float = 0.15):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate == 0:
            return patches
        mask = (torch.rand(patches.size(0), patches.size(1), 1, device=patches.device) > self.drop_rate).float()
        return (patches * mask) / (1.0 - self.drop_rate)
