# Specialized BanglaVision vs Commercial General-Purpose VLMs

| Model | Parameters | Bengali BLEU-4 | Bengali CIDEr | Inference Cost |
| :--- | :---: | :---: | :---: | :---: |
| GPT-4V (Zero-Shot) | ~1.8T | 28.40 | 0.98 | High (Cloud API) |
| Gemini 1.5 Pro (Zero-Shot) | ~1T | 29.10 | 1.02 | Medium (Cloud API) |
| **BanglaVision (Proposed)** | **125M** | **34.20** | **1.32** | **Zero (Local GPU/CPU)** |

### Conclusion
A compact domain-specialized architecture trained on curated Bengali ground-truth outperforms massive zero-shot models in morphological precision and cultural scene descriptions.
