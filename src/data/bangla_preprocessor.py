import re
import unicodedata

class BanglaTextNormalizer:
    """Normalizes and cleans Bengali caption text."""
    def __init__(self):
        # Bengali punctuation and characters
        self.bengali_digits = "০১২৩৪৫৬৭৮৯"
        self.whitespace_regex = re.compile(r'\s+')
        self.punctuation_regex = re.compile(r'[\.,!?;:\|।॥"']')

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # Unicode normalization (NFC)
        text = unicodedata.normalize('NFC', text)
        # Remove extra whitespace
        text = self.whitespace_regex.sub(' ', text).strip()
        # Clean special symbols while preserving Bengali script range (ঀ-৿)
        cleaned = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s\.,\|\u0964]', '', text)
        return cleaned.strip()
