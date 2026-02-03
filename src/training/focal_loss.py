import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCrossEntropyLoss(nn.Module):
    """Focal loss down-weighting easy tokens to focus gradient on hard Bengali vocabulary words."""
    def __init__(self, gamma: float = 2.0, pad_idx: int = 0):
        super().__init__()
        self.gamma = gamma
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.pad_idx
        logits_flat = logits.view(-1, logits.size(-1))[mask.view(-1)]
        targets_flat = targets.view(-1)[mask.view(-1)]

        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
