import torch.optim as optim

def build_optimizer(model, base_lr: float = 1e-4, encoder_lr: float = 1e-5, weight_decay: float = 1e-4):
    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n and p.requires_grad]

    param_groups = [
        {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': decoder_params, 'lr': base_lr, 'weight_decay': weight_decay}
    ]
    return optim.AdamW(param_groups)
