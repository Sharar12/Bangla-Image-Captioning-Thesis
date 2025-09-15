import torch
import torch.nn.functional as F
from typing import List, Tuple

class BeamSearchCaptioner:
    def __init__(self, model, vocab, beam_width: int = 5, max_len: int = 30, length_penalty_alpha: float = 0.7, rep_penalty: float = 1.2):
        self.model = model
        self.vocab = vocab
        self.beam_width = beam_width
        self.max_len = max_len
        self.alpha = length_penalty_alpha
        self.rep_penalty = rep_penalty
        self.sos_idx = vocab.stoi.get("<SOS>", 1)
        self.eos_idx = vocab.stoi.get("<EOS>", 2)

    def search(self, image_tensor: torch.Tensor) -> List[Tuple[str, float]]:
        self.model.eval()
        with torch.no_grad():
            features = self.model.encoder(image_tensor)
            h = self.model.decoder.init_hidden_state(features)

            # (sequence, score, hidden)
            beams = [([self.sos_idx], 0.0, h)]
            completed_beams = []

            for _ in range(self.max_len):
                candidates = []
                for seq, score, hid in beams:
                    if seq[-1] == self.eos_idx:
                        completed_beams.append((seq, score))
                        continue

                    last_token = torch.tensor([seq[-1]], device=image_tensor.device)
                    embedded = self.model.decoder.embedding(last_token)
                    ctx, _ = self.model.decoder.attention(features, hid)
                    new_hid = self.model.decoder.gru(torch.cat([embedded, ctx], dim=1), hid)
                    logits = self.model.decoder.fc_out(new_hid)

                    # Repetition penalty
                    for prev_tok in set(seq):
                        logits[0, prev_tok] /= self.rep_penalty

                    log_probs = F.log_softmax(logits, dim=-1)
                    topk_probs, topk_indices = log_probs.topk(self.beam_width)

                    for prob, idx in zip(topk_probs[0], topk_indices[0]):
                        candidates.append((seq + [idx.item()], score + prob.item(), new_hid))

                if not candidates:
                    break

                # Sort by score normalized by length
                candidates.sort(key=lambda x: x[1] / (len(x[0]) ** self.alpha), reverse=True)
                beams = candidates[:self.beam_width]

            if not completed_beams:
                completed_beams = [(b[0], b[1]) for b in beams]

            completed_beams.sort(key=lambda x: x[1] / (len(x[0]) ** self.alpha), reverse=True)
            results = []
            for seq, score in completed_beams[:self.beam_width]:
                words = [self.vocab.itos.get(idx, "<UNK>") for idx in seq if idx not in (self.sos_idx, self.eos_idx)]
                results.append((" ".join(words), score))
            return results
