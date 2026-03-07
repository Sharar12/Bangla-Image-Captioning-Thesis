import unittest
from src.training.scst_trainer import SCSTTrainer

class TestSCST(unittest.TestCase):
    def test_instantiation(self):
        trainer = SCSTTrainer(model=None, optimizer=None, cider_scorer=None)
        self.assertIsNotNone(trainer)

if __name__ == '__main__':
    unittest.main()
