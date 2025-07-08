import json
import random
from typing import List, Dict

def split_dataset(annotations: List[Dict], train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    shuffled = annotations.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }

if __name__ == "__main__":
    print("Hybrid dataset split utility ready for 60K multi-source Bangla dataset.")
