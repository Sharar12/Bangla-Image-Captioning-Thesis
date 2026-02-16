from torch.optim.lr_scheduler import _LRScheduler

class SWALRScheduler(_LRScheduler):
    def __init__(self, optimizer, swa_lr=0.00005, anneal_epochs=5, last_epoch=-1):
        self.swa_lr = swa_lr
        self.anneal_epochs = anneal_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.swa_lr for _ in self.optimizer.param_groups]
