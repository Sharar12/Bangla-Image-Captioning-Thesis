import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """Contrastive InfoNCE loss aligning visual patch tokens with Bengali sentence embeddings."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        img_norm = F.normalize(img_embeds, dim=-1)
        txt_norm = F.normalize(text_embeds, dim=-1)
        sim = torch.matmul(img_norm, txt_norm.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
        return loss
