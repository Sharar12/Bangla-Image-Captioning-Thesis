import unittest
from src.data.bangla_preprocessor import BanglaTextNormalizer

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.normalizer = BanglaTextNormalizer()

    def test_normalize_clean_text(self):
        raw = "  একটি   সুন্দর   দৃশ্য!  "
        clean = self.normalizer.normalize(raw)
        self.assertEqual(clean, "একটি সুন্দর দৃশ্য!")

    def test_empty_string_handling(self):
        self.assertEqual(self.normalizer.normalize(""), "")
        self.assertEqual(self.normalizer.normalize(None), "")

if __name__ == '__main__':
    unittest.main()
