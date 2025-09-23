import torch

def clip_gradients(model, max_norm: float = 5.0):
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()
