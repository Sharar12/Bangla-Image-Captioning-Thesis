class LengthNormalizedEarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.counter = 0

    def should_terminate(self, best_score: float, current_score: float) -> bool:
        if current_score < best_score * 0.8:
            self.counter += 1
        return self.counter >= self.patience
