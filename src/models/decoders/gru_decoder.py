import torch
import torch.nn as nn
import random
from src.models.decoders.base_decoder import BaseDecoder
from src.models.layers.bahdanau_attention import BahdanauAttention

class AttentionDecoderGRU(BaseDecoder):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, encoder_dim: int,
                 attention_dim: int = 256, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__(vocab_size, embed_dim, hidden_dim, pad_idx)
        self.encoder_dim = encoder_dim
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        
        self.init_hidden = nn.Linear(encoder_dim, hidden_dim)
        self.gru = nn.GRUCell(embed_dim + encoder_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> torch.Tensor:
        mean_enc = encoder_out.mean(dim=1)
        return torch.tanh(self.init_hidden(mean_enc))

    def forward(self, features: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        batch_size = features.size(0)
        seq_len = captions.size(1)
        hidden = self.init_hidden_state(features)

        # Pre-project visual features for faster attention computation across sequence
        proj_features = self.attention.W_encoder(features)

        outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device=features.device)
        input_token = captions[:, 0]

        for t in range(seq_len):
            embedded = self.dropout(self.embedding(input_token))
            
            # Efficient attention using cached projection
            proj_hidden = self.attention.W_decoder(hidden).unsqueeze(1)
            scores = self.attention.V_attention(torch.tanh(proj_features + proj_hidden)).squeeze(-1)
            alpha = torch.softmax(scores, dim=-1)
            context = (features * alpha.unsqueeze(-1)).sum(dim=1)

            gru_input = torch.cat([embedded, context], dim=1)
            hidden = self.gru(gru_input, hidden)
            logits = self.fc_out(hidden)
            outputs[:, t, :] = logits

            is_teacher = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = captions[:, t] if (is_teacher and t < seq_len - 1) else top1

        return outputs
