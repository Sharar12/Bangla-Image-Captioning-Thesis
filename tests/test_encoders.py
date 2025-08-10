import unittest
import torch
from src.models.encoders.resnet_encoder import ResNetEncoder

class TestEncoders(unittest.TestCase):
    def test_resnet_forward_shape(self):
        encoder = ResNetEncoder(embed_dim=256)
        dummy_img = torch.zeros(2, 3, 224, 224)
        out = encoder(dummy_img)
        self.assertEqual(out.shape, (2, 49, 256))

if __name__ == '__main__':
    unittest.main()
