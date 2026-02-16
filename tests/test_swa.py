import unittest
import torch
from src.training.swa_scheduler import SWALRScheduler

class TestSWA(unittest.TestCase):
    def test_swa_lr(self):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        sched = SWALRScheduler(opt, swa_lr=0.001)
        self.assertEqual(sched.get_lr(), [0.001])

if __name__ == '__main__':
    unittest.main()
