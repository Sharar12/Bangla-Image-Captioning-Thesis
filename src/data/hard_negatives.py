import random

def generate_hard_negative_captions(caption: str, vocab) -> str:
    tokens = caption.split()
    if len(tokens) > 2:
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = "গাছ" if tokens[idx] != "গাছ" else "নদী"
    return " ".join(tokens)
