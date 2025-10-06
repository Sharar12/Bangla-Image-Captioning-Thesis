import unittest
from src.evaluation.metrics_suite import CaptionMetricsSuite

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.suite = CaptionMetricsSuite()
        self.references = [
            [["একটি", "লাল", "গাড়ি", "রাস্তায়", "দাঁড়িয়ে", "আছে"], ["রাস্তায়", "একটি", "লাল", "গাড়ি"]]
        ]
        self.hypotheses = [
            ["একটি", "লাল", "গাড়ি", "রাস্তায়", "আছে"]
        ]

    def test_metrics_computation(self):
        results = self.suite.evaluate_all(self.references, self.hypotheses)
        self.assertIn("BLEU-1", results)
        self.assertIn("BLEU-4", results)
        self.assertIn("ROUGE-L", results)
        self.assertIn("METEOR", results)
        self.assertIn("CIDEr", results)
        self.assertGreater(results["BLEU-1"], 0.0)

if __name__ == '__main__':
    unittest.main()
