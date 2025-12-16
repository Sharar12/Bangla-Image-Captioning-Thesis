import torch

def filter_high_confidence_predictions(hypotheses: list, min_score: float = -1.5):
    # Retain captions where log-likelihood exceeds threshold
    return [h for h in hypotheses if h[1] >= min_score]
