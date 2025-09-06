import unittest
import torch
from src.models.decoders.gru_decoder import AttentionDecoderGRU
from src.models.decoders.lstm_decoder import LSTMDecoderWithAttention

class TestDecoders(unittest.TestCase):
    def test_gru_forward(self):
        decoder = AttentionDecoderGRU(vocab_size=100, embed_dim=64, hidden_dim=128, encoder_dim=256)
        feats = torch.randn(2, 49, 256)
        captions = torch.randint(0, 100, (2, 15))
        out = decoder(feats, captions)
        self.assertEqual(out.shape, (2, 15, 100))

    def test_lstm_forward(self):
        decoder = LSTMDecoderWithAttention(vocab_size=100, embed_dim=64, hidden_dim=128, encoder_dim=256)
        feats = torch.randn(2, 49, 256)
        captions = torch.randint(0, 100, (2, 15))
        out = decoder(feats, captions)
        self.assertEqual(out.shape, (2, 15, 100))

if __name__ == '__main__':
    unittest.main()
