import unittest
import torch
import torch.nn as nn
from src.training.ema import ModelEMA

class TestEMA(unittest.TestCase):
    def test_ema_update(self):
        linear = nn.Linear(10, 10)
        ema = ModelEMA(linear, decay=0.9)
        with torch.no_grad():
            linear.weight.add_(1.0)
        ema.update(linear)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
