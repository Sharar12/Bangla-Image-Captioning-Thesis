import unittest
from src.inference.diverse_beam_search import DiverseBeamSearchDecoder

class TestDiverseBeam(unittest.TestCase):
    def test_instantiation(self):
        decoder = DiverseBeamSearchDecoder(model=None, vocab=None, num_groups=2, beams_per_group=2)
        self.assertEqual(decoder.num_groups, 2)
        self.assertEqual(decoder.beams_per_group, 2)

if __name__ == '__main__':
    unittest.main()
