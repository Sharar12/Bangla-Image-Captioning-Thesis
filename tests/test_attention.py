import unittest
import torch
from src.models.layers.bahdanau_attention import BahdanauAttention

class TestAttention(unittest.TestCase):
    def test_attention_weights_sum_to_one(self):
        att = BahdanauAttention(encoder_dim=256, decoder_dim=256, attention_dim=128)
        enc_out = torch.randn(2, 49, 256)
        dec_hid = torch.randn(2, 256)
        context, alpha = att(enc_out, dec_hid)

        self.assertEqual(context.shape, (2, 256))
        self.assertEqual(alpha.shape, (2, 49))
        self.assertTrue(torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-5))

if __name__ == '__main__':
    unittest.main()
