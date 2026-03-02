import torch
import torch.nn as nn

class MultiTaskTagHead(nn.Module):
    """Predicts salient Bengali object tags concurrently with autoregressive caption generation."""
    def __init__(self, visual_dim: int, num_tags: int = 500):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_tags),
            nn.Sigmoid()
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        pooled = visual_features.mean(dim=1)
        return self.classifier(pooled)
