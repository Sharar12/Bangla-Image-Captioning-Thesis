from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict

class BLEUScorer:
    """Calculates BLEU-1 to BLEU-4 scores for Bangla generated captions against references."""
    def __init__(self):
        self.smooth = SmoothingFunction().method1

    def compute_score(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
        n = max(1, len(hypotheses))

        for refs, hyp in zip(references, hypotheses):
            bleu1 += sentence_bleu(refs, hyp, weights=(1.0, 0, 0, 0), smoothing_function=self.smooth)
            bleu2 += sentence_bleu(refs, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smooth)
            bleu3 += sentence_bleu(refs, hyp, weights=(0.333, 0.333, 0.333, 0), smoothing_function=self.smooth)
            bleu4 += sentence_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth)

        return {
            "BLEU-1": (bleu1 / n) * 100.0,
            "BLEU-2": (bleu2 / n) * 100.0,
            "BLEU-3": (bleu3 / n) * 100.0,
            "BLEU-4": (bleu4 / n) * 100.0,
        }
