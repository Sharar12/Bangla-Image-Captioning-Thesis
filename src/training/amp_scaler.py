import torch
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled)

    def step(self, loss, optimizer, max_norm: float = 5.0):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad()
