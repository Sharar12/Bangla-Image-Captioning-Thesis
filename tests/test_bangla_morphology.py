import unittest
from src.data.bangla_morphology import BengaliMorphologyHandler

class TestBengaliMorphology(unittest.TestCase):
    def test_grapheme_handler(self):
        handler = BengaliMorphologyHandler()
        res = handler.segment_grapheme_clusters("বিজ্ঞান")
        self.assertGreater(len(res), 0)

if __name__ == '__main__':
    unittest.main()
