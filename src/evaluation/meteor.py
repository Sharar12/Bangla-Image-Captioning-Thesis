from typing import List, Dict

class METEORScorer:
    """METEOR alignment metric approximation for Bengali captions."""
    def compute_score(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        total_meteor = 0.0
        n = max(1, len(hypotheses))

        for refs, hyp in zip(references, hypotheses):
            best_score = 0.0
            hyp_set = set(hyp)
            for ref in refs:
                ref_set = set(ref)
                overlap = len(hyp_set & ref_set)
                if overlap == 0:
                    continue
                prec = overlap / max(1, len(hyp))
                rec = overlap / max(1, len(ref))
                f_mean = (10 * prec * rec) / (9 * prec + rec) if (9 * prec + rec) > 0 else 0
                penalty = 0.5 * (1 - (overlap / max(1, len(hyp))))
                score = f_mean * (1 - penalty)
                if score > best_score:
                    best_score = score
            total_meteor += best_score

        return {"METEOR": (total_meteor / n) * 100.0}
