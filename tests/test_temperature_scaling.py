import unittest
import torch
from src.models.layers.temperature_scaling import CrossAttentionTemperatureScaler

class TestTemperatureScaling(unittest.TestCase):
    def test_forward_scaling(self):
        scaler = CrossAttentionTemperatureScaler(init_temperature=2.0)
        scores = torch.tensor([[4.0, 2.0]])
        out = scaler(scores)
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 1.0]])))

if __name__ == '__main__':
    unittest.main()
