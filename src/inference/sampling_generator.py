import torch
import torch.nn.functional as F

def sample_top_k_top_p(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> int:
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_val = values[:, -1]
        logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(float('-inf'), device=logits.device), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token
