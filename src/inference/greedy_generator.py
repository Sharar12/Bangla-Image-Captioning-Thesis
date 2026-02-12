import torch

def generate_greedy(model, image_tensor: torch.Tensor, vocab, max_len: int = 30, sos_idx: int = 1, eos_idx: int = 2):
    """Optimized greedy decoding with pre-computed visual context caching."""
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        hidden = model.decoder.init_hidden_state(features) if hasattr(model.decoder, 'init_hidden_state') else None

        current_token = torch.tensor([sos_idx], device=image_tensor.device)
        caption_indices = [sos_idx]

        for _ in range(max_len):
            embedded = model.decoder.embedding(current_token)
            context, _ = model.decoder.attention(features, hidden)
            gru_in = torch.cat([embedded, context], dim=1)
            hidden = model.decoder.gru(gru_in, hidden)
            logits = model.decoder.fc_out(hidden)
            next_token = logits.argmax(1).item()

            caption_indices.append(next_token)
            if next_token == eos_idx:
                break
            current_token = torch.tensor([next_token], device=image_tensor.device)

        words = [vocab.itos.get(idx, "<UNK>") for idx in caption_indices if idx not in (sos_idx, eos_idx)]
        return " ".join(words)
