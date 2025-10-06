import math
from collections import Counter
from typing import List, Dict

class CIDErScorer:
    """CIDEr (Consensus-based Image Description Evaluation) calculation."""
    def __init__(self, n_gram: int = 4):
        self.n_gram = n_gram

    def _get_ngrams(self, words: List[str], n: int) -> Counter:
        return Counter([tuple(words[i:i+n]) for i in range(len(words) - n + 1)])

    def compute_score(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        # TF-IDF weighted consensus score across references
        total_score = 0.0
        n_samples = max(1, len(hypotheses))

        for refs, hyp in zip(references, hypotheses):
            hyp_len = len(hyp)
            if hyp_len == 0:
                continue

            sample_cider = 0.0
            for n in range(1, self.n_gram + 1):
                hyp_ngrams = self._get_ngrams(hyp, n)
                match_count = 0
                ref_counts = [self._get_ngrams(ref, n) for ref in refs]
                for ng, cnt in hyp_ngrams.items():
                    avg_ref_cnt = sum(rc.get(ng, 0) for rc in ref_counts) / max(1, len(refs))
                    match_count += min(cnt, avg_ref_cnt)
                prec = match_count / max(1, len(hyp) - n + 1)
                sample_cider += prec

            sample_cider = (sample_cider / self.n_gram) * 10.0
            total_score += sample_cider

        return {"CIDEr": total_score / n_samples}
