from collections import Counter
import numpy as np

def compute_corpus_statistics(captions):
    lengths = [len(c.split()) for c in captions]
    vocab = Counter([word for c in captions for word in c.split()])

    return {
        "total_captions": len(captions),
        "unique_tokens": len(vocab),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "p95_length": float(np.percentile(lengths, 95)),
        "max_length": int(np.max(lengths))
    }
