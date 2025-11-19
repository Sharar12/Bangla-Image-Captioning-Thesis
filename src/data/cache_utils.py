import os
import torch

class ImageFeatureCache:
    def __init__(self, cache_dir: str = "cache/features"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, image_id: str):
        path = os.path.join(self.cache_dir, f"{image_id}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return None

    def put(self, image_id: str, tensor: torch.Tensor):
        path = os.path.join(self.cache_dir, f"{image_id}.pt")
        torch.save(tensor.cpu(), path)
