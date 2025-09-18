import torch

def evaluate_validation_loss(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            logits = model(images, captions, teacher_forcing_ratio=0.0)
            loss = criterion(logits[:, :-1, :], captions[:, 1:])
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))
