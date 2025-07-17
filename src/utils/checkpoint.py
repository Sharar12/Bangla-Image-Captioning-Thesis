import os
import torch

class CheckpointManager:
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_bleu = 0.0

    def save(self, model, optimizer, epoch: int, bleu_score: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bleu_score': bleu_score
        }
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pt")
        torch.save(state, latest_path)

        if is_best or bleu_score > self.best_bleu:
            self.best_bleu = bleu_score
            best_path = os.path.join(self.save_dir, f"best_model_bleu_{bleu_score:.4f}.pt")
            torch.save(state, best_path)
            print(f"[Checkpoint] New best model saved with BLEU: {bleu_score:.4f}")
