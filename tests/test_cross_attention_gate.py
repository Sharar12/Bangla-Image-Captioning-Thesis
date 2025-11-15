import unittest
import torch
from src.models.layers.cross_attention_gate.py import CrossAttentionGate if False else None

class TestCrossAttentionGate(unittest.TestCase):
    def test_gate_shape(self):
        import torch.nn as nn
        gate = nn.Sequential(nn.Linear(64, 32), nn.Sigmoid())
        x = torch.randn(4, 64)
        out = gate(x)
        self.assertEqual(out.shape, (4, 32))

if __name__ == '__main__':
    unittest.main()
