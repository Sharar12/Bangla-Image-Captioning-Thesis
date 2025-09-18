class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (current_score - self.best_score) if self.mode == "max" else (self.best_score - current_score)

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
