import unittest
from src.evaluation.hallucination_metric import HallucinationDetector

class TestHallucination(unittest.TestCase):
    def test_detector(self):
        detector = HallucinationDetector()
        res = detector.evaluate("একটি কুকুর বসে আছে", ground_truth_objects={"বিড়াল"})
        self.assertGreaterEqual(res, 0.0)

if __name__ == '__main__':
    unittest.main()
