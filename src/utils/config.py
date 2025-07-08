from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExperimentConfig:
    # Model parameters
    encoder_name: str = "google/siglip-base-patch16-224"
    decoder_type: str = "hybrid_gru_lstm"
    embed_size: int = 512
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3

    # Training hyperparams
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 1e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-4
    seed: int = 42

    # Hardware
    device: str = "cuda"
    fp16: bool = True
