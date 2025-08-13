import torch.nn as nn

class EncoderFreezer:
    @staticmethod
    def freeze(encoder: nn.Module):
        for param in encoder.parameters():
            param.requires_grad = False
        print("[Encoder] All visual encoder layers frozen.")

    @staticmethod
    def unfreeze_top_k_layers(encoder: nn.Module, k: int = 2):
        # Unfreeze top-k transformer layers
        for param in encoder.parameters():
            param.requires_grad = True
        print(f"[Encoder] Top {k} encoder layers unfrozen for fine-tuning.")
