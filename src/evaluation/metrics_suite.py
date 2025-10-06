from typing import List, Dict
from src.evaluation.bleu import BLEUScorer
from src.evaluation.rouge import ROUGELScorer
from src.evaluation.meteor import METEORScorer
from src.evaluation.cider import CIDErScorer

class CaptionMetricsSuite:
    """Comprehensive metric evaluation suite covering BLEU, ROUGE-L, METEOR, and CIDEr."""
    def __init__(self):
        self.bleu = BLEUScorer()
        self.rouge = ROUGELScorer()
        self.meteor = METEORScorer()
        self.cider = CIDErScorer()

    def evaluate_all(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        results = {}
        results.update(self.bleu.compute_score(references, hypotheses))
        results.update(self.rouge.compute_score(references, hypotheses))
        results.update(self.meteor.compute_score(references, hypotheses))
        results.update(self.cider.compute_score(references, hypotheses))
        return results
