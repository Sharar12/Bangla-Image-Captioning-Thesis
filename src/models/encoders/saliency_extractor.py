import torch

def extract_visual_saliency(features: torch.Tensor) -> torch.Tensor:
    # Feature magnitude as saliency proxy: (B, N)
    saliency = torch.norm(features, dim=-1)
    saliency = (saliency - saliency.min(dim=-1, keepdim=True)[0]) / (saliency.max(dim=-1, keepdim=True)[0] - saliency.min(dim=-1, keepdim=True)[0] + 1e-8)
    return saliency
