import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.encoders.resnet_encoder import ResNet50Encoder
from src.models.decoders.gru_decoder import AttentionDecoderGRU
from src.models.encoder_decoder import BanglaCaptioningModel
from src.training.trainer import CaptionTrainer
from src.training.losses import LabelSmoothingLoss

class TestTrainingPipeline(unittest.TestCase):
    def test_single_train_step(self):
        encoder = ResNet50Encoder(pretrained=False)
        decoder = AttentionDecoderGRU(vocab_size=50, embed_dim=32, hidden_dim=64, encoder_dim=2048)
        model = BanglaCaptioningModel(encoder, decoder)

        images = torch.randn(4, 3, 224, 224)
        captions = torch.randint(0, 50, (4, 10))
        dataset = TensorDataset(images, captions)
        loader = DataLoader(dataset, batch_size=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = LabelSmoothingLoss(smoothing=0.1)
        trainer = CaptionTrainer(model, optimizer, criterion)

        loss = trainer.train_epoch(loader, teacher_forcing_ratio=0.5)
        self.assertGreater(loss, 0.0)

if __name__ == '__main__':
    unittest.main()
