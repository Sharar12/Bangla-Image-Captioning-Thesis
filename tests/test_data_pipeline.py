import unittest
import torch
from src.data.vocabulary import Vocabulary
from src.data.dataloader import CaptionCollate

class TestDataPipeline(unittest.TestCase):
    def test_vocabulary_encoding(self):
        vocab = Vocabulary(freq_threshold=1)
        corpus = ["একটি বিড়াল", "একটি কুকুর"]
        vocab.build_vocabulary(corpus)

        self.assertIn("একটি", vocab.stoi)
        self.assertEqual(vocab.stoi["<PAD>"], 0)
        self.assertEqual(vocab.stoi["<SOS>"], 1)

    def test_collate_padding(self):
        collate = CaptionCollate(pad_idx=0)
        t1 = (torch.zeros(3, 224, 224), torch.tensor([1, 4, 5, 2]))
        t2 = (torch.zeros(3, 224, 224), torch.tensor([1, 4, 2]))
        imgs, targets = collate([t1, t2])

        self.assertEqual(targets.shape[1], 4)
        self.assertEqual(targets[1, 3].item(), 0) # padded

if __name__ == '__main__':
    unittest.main()
