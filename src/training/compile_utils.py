import torch

def compile_model_if_available(model):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    return model
