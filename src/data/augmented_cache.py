import os
import torch

class FeatureCache:
    def __init__(self, cache_dir: str = ".cache/features"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save(self, img_id: str, feature_tensor: torch.Tensor):
        path = os.path.join(self.cache_dir, f"{img_id}.pt")
        torch.save(feature_tensor.cpu(), path)

    def load(self, img_id: str):
        path = os.path.join(self.cache_dir, f"{img_id}.pt")
        return torch.load(path) if os.path.exists(path) else None
