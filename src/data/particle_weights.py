from collections import Counter
from typing import List, Dict

BENGALI_PARTICLES = {"টি", "টা", "খানা", "খানি", "গুলো", "গুলি", "রা", "দের"}

def compute_particle_loss_weights(vocab_stoi: Dict[str, int], smoothing: float = 0.5) -> Dict[int, float]:
    weights = {}
    for particle in BENGALI_PARTICLES:
        if particle in vocab_stoi:
            weights[vocab_stoi[particle]] = 1.0 + smoothing
    return weights
