import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq
from torchvision import transforms
from PIL import Image

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL_FOLDER))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MAX_SEQ_LEN = 32

MODEL_NAME = "microsoft/Florence-2-base"
HF_TOKEN = "YOUR_HF_TOKEN"

FEATURE_DIM = 1024
EMBED_DIM = 1024
NUM_LAYERS = 10
HIDDEN_DIM = 4096
NHEAD = 16
DROPOUT = 0.3


class Vocabulary:
    def __init__(self, word2idx=None):
        if word2idx:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}
            self.n_words = len(word2idx)
        else:
            self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
            self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            self.n_words = 4

    def decode(self, tokens):
        words = []
        for t in tokens:
            w = self.idx2word.get(t, "<UNK>")
            if w in ["<EOS>", "<PAD>"]:
                break
            if w not in ["<SOS>", "<UNK>"]:
                words.append(w)
        return " ".join(words)


def load_captions(caption_file):
    captions = {}
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                captions[parts[0]] = parts[1]
    return captions


def calculate_rouge_l(reference, hypothesis):
    ref_set = set(reference)
    hyp_set = set(hypothesis)
    if len(ref_set) == 0 or len(hyp_set) == 0:
        return 0.0
    overlap = len(ref_set.intersection(hyp_set))
    precision = overlap / len(hyp_set) if len(hyp_set) > 0 else 0
    recall = overlap / len(ref_set) if len(ref_set) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(reference, hypothesis):
    reference = reference.split()
    hypothesis = hypothesis.split()
    smoothing = SmoothingFunction().method1
    bleu1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    try:
        meteor = meteor_score([reference], hypothesis)
    except:
        meteor = 0.0
    rouge_l = calculate_rouge_l(reference, hypothesis)
    return {'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4, 'meteor': meteor, 'rouge_l': rouge_l}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Florence2ForConditionalGeneration
        full_model = Florence2ForConditionalGeneration.from_pretrained(
            MODEL_NAME, 
            token=HF_TOKEN,
            ignore_mismatched_sizes=True
        )
        if hasattr(full_model, 'vision_tower'):
            self.vision_tower = full_model.vision_tower
        elif hasattr(full_model, 'model') and hasattr(full_model.model, 'vision_tower'):
            self.vision_tower = full_model.model.vision_tower
        else:
            self.full_model = full_model
            self.vision_tower = None
        
        if self.vision_tower is not None:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
        self.feature_dim = 1024

    def forward(self, pixel_values):
        with torch.no_grad():
            if self.vision_tower is not None:
                outputs = self.vision_tower(pixel_values)
            else:
                outputs = self.full_model(pixel_values, labels=None)
        return outputs.last_hidden_state


class Decoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab_size=10000, feature_dim=FEATURE_DIM, num_layers=NUM_LAYERS, nhead=NHEAD, dropout=DROPOUT):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.word2idx = None
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_feature = nn.Linear(feature_dim, embed_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def greedy_decode(self, features, max_len=MAX_SEQ_LEN):
        batch_size = features.size(0)
        
        memory = self.fc_feature(features[:, 0:1, :])
        memory = self.encoder(memory)
        
        generated = torch.full((batch_size, 1), self.word2idx["<SOS>"], dtype=torch.long).to(features.device)
        
        for _ in range(max_len - 1):
            embed = self.embedding(generated) * (self.embed_dim ** 0.5)
            embed = embed + self.pos_encoder[:, :generated.size(1), :]
            
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(features.device)
            decoder_output = self.decoder(embed, memory, tgt_mask=tgt_mask)
            out = self.fc_out(decoder_output)
            
            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == self.word2idx["<EOS>"]).all():
                break
        
        return generated


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, word2idx):
        super().__init__()
        self.word2idx = word2idx
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=vocab_size)
        self.decoder.word2idx = word2idx


def test_image(image_path):
    print(f"Device: {DEVICE}, Testing: {image_path}")
    
    captions = load_captions(os.path.join(DATA_DIR, "caption.txt"))
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "final_model.pt"), map_location=DEVICE)
    word2idx = checkpoint["vocab"]
    vocab = Vocabulary(word2idx)
    
    model = CaptionModel(vocab.n_words, word2idx).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert("RGB")
    pixel_values = transform(img).unsqueeze(0).to(DEVICE)
    
    img_name = os.path.basename(image_path)
    actual_caption = captions.get(img_name, "No caption")
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            features = model.encoder(pixel_values)
            predictions = model.decoder.greedy_decode(features, MAX_SEQ_LEN)
    
    generated = vocab.decode(predictions[0].cpu().numpy())
    
    print(f"\nGenerated: {generated}")
    print(f"Actual: {actual_caption}")
    
    if actual_caption != "No caption":
        scores = calculate_metrics(actual_caption, generated)
        print(f"BLEU-4: {scores['bleu4']:.4f}, METEOR: {scores['meteor']:.4f}")
    
    img.save(os.path.join(OUTPUT_DIR, "sample_output.jpg"))
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'sample_output.jpg')}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(DATA_DIR, "Pictures", "6001.jpg")
    test_image(image_path)
