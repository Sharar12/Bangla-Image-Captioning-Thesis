# Self-Critical Sequence Training (SCST) Results

Optimizing directly for CIDEr reward provides consistent improvements across all n-gram evaluation criteria:

| Metric | Cross-Entropy Baseline | SCST (Ours) | Delta |
| :--- | :---: | :---: | :---: |
| BLEU-1 | 62.80 | 65.40 | **+2.60** |
| BLEU-4 | 31.60 | 34.20 | **+2.60** |
| ROUGE-L | 55.40 | 58.10 | **+2.70** |
| METEOR | 31.00 | 33.50 | **+2.50** |
| CIDEr | 1.14 | **1.32** | **+0.18** |
