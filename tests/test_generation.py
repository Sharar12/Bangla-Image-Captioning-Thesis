import unittest
import torch
from src.inference.sampling_generator import sample_top_k_top_p

class TestGeneration(unittest.TestCase):
    def test_top_k_sampling_bounds(self):
        logits = torch.randn(1, 100)
        idx = sample_top_k_top_p(logits, top_k=10, top_p=0.9, temperature=0.8)
        self.assertTrue(0 <= idx < 100)

if __name__ == '__main__':
    unittest.main()
