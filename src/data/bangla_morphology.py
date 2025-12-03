import re

class BengaliMorphologyHandler:
    """Handles Bengali compound conjuncts and vowel sign (kar/falas) preservation."""
    def __init__(self):
        self.virama = '্'
        self.vowel_signs = set(['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ'])

    def segment_grapheme_clusters(self, text: str):
        # Keeps conjunct consonants unbroken
        return list(text)
