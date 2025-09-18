import os
import glob
import torch
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3, metric_mode: str = "max"):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.metric_mode = metric_mode
        self.history = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, state: Dict[str, Any], metric_val: float, epoch: int):
        ckpt_name = f"ckpt_epoch_{epoch:03d}_metric_{metric_val:.4f}.pt"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        torch.save(state, ckpt_path)
        self.history.append((metric_val, ckpt_path))

        reverse = (self.metric_mode == "max")
        self.history.sort(key=lambda x: x[0], reverse=reverse)

        while len(self.history) > self.max_to_keep:
            worst_metric, worst_path = self.history.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)

        # Always save latest
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(state, latest_path)
        return ckpt_path
