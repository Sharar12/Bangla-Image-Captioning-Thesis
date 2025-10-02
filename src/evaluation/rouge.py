from typing import List, Dict

class ROUGELScorer:
    """Computes ROUGE-L longest common subsequence F1 score."""
    def _lcs(self, a: List[str], b: List[str]) -> int:
        lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
        return lengths[-1][-1]

    def compute_score(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        total_f1 = 0.0
        n = max(1, len(hypotheses))

        for refs, hyp in zip(references, hypotheses):
            best_f1 = 0.0
            for ref in refs:
                lcs_len = self._lcs(ref, hyp)
                if lcs_len == 0 or len(hyp) == 0 or len(ref) == 0:
                    continue
                prec = lcs_len / len(hyp)
                rec = lcs_len / len(ref)
                f1 = (2 * prec * rec) / (prec + rec)
                if f1 > best_f1:
                    best_f1 = f1
            total_f1 += best_f1

        return {"ROUGE-L": (total_f1 / n) * 100.0}
