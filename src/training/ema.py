import torch
import copy

class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)
