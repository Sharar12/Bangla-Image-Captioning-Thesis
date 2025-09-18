import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class CaptionTrainer:
    def __init__(self, model, optimizer, criterion, scaler: GradScaler = None, grad_accum_steps: int = 1, clip_grad: float = 5.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler or GradScaler(enabled=torch.cuda.is_available())
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad = clip_grad

    def train_epoch(self, dataloader, teacher_forcing_ratio: float = 0.5):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (images, captions) in enumerate(tqdm(dataloader, desc="Training")):
            device = next(self.model.parameters()).device
            images, captions = images.to(device), captions.to(device)

            with autocast(enabled=torch.cuda.is_available()):
                logits = self.model(images, captions, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = self.criterion(logits[:, :-1, :], captions[:, 1:])
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps

        return total_loss / len(dataloader)
