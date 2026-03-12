class DynamicLossWeightScheduler:
    def __init__(self, ce_start: float = 1.0, rl_start: float = 0.0, transition_epoch: int = 10):
        self.ce_start = ce_start
        self.rl_start = rl_start
        self.transition_epoch = transition_epoch

    def get_weights(self, epoch: int):
        if epoch < self.transition_epoch:
            return 1.0, 0.0
        alpha = min(1.0, (epoch - self.transition_epoch) / 10.0)
        return (1.0 - alpha), alpha
