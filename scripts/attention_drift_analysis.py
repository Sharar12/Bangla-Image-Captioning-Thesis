import numpy as np

def calculate_attention_drift(attention_weights_seq):
    """Quantifies whether attention focuses on progressively diverse spatial areas."""
    centroids = []
    for att in attention_weights_seq:
        grid = att.reshape(7, 7)
        cy, cx = np.unravel_index(np.argmax(grid), (7, 7))
        centroids.append((cx, cy))
    
    shifts = [np.linalg.norm(np.array(centroids[i]) - np.array(centroids[i-1])) for i in range(1, len(centroids))]
    return float(np.mean(shifts)) if shifts else 0.0
