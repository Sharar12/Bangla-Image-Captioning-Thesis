from collections import Counter
from typing import List

def calculate_ngram_diversity(captions: List[str], n: int = 2) -> float:
    """Measures Distinct-N metric (ratio of distinct n-grams to total n-grams)."""
    all_ngrams = []
    for cap in captions:
        tokens = cap.split()
        all_ngrams.extend([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)
