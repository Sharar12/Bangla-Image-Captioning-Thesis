import math

class TeacherForcingScheduler:
    """Decays teacher forcing ratio across training epochs."""
    def __init__(self, initial_ratio: float = 1.0, min_ratio: float = 0.1, decay_type: str = "exponential", decay_rate: float = 0.05):
        self.initial_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.decay_type = decay_type
        self.decay_rate = decay_rate

    def get_ratio(self, epoch: int) -> float:
        if self.decay_type == "linear":
            ratio = self.initial_ratio - (epoch * self.decay_rate)
        elif self.decay_type == "exponential":
            ratio = self.initial_ratio * math.exp(-self.decay_rate * epoch)
        elif self.decay_type == "cosine":
            ratio = self.min_ratio + 0.5 * (self.initial_ratio - self.min_ratio) * (1 + math.cos(epoch * math.pi / 50))
        else:
            ratio = self.initial_ratio
        return max(self.min_ratio, min(self.initial_ratio, ratio))
