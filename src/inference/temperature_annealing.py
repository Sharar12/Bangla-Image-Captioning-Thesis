class TemperatureAnnealer:
    def __init__(self, initial_temp: float = 1.0, min_temp: float = 0.6, decay: float = 0.95):
        self.temp = initial_temp
        self.min_temp = min_temp
        self.decay = decay

    def step(self) -> float:
        self.temp = max(self.min_temp, self.temp * self.decay)
        return self.temp
