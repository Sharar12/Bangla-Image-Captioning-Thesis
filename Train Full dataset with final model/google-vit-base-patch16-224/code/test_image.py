import os
import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
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
MODEL_DIR = os.path.join(MODEL_FOLDER, "model")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MAX_SEQ_LEN = 32


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
            parts = line.split("   ")
            if len(parts) >= 2:
                img_name = parts[0]
                caption = parts[1]
                if img_name not in captions:
                    captions[img_name] = caption
            elif len(line.split()) >= 2:
                parts2 = line.split(None, 1)
                img_name = parts2[0]
                caption = parts2[1]
                if img_name not in captions:
                    captions[img_name] = caption
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
        self.vit = ViTModel.from_pretrained(MODEL_DIR)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.feature_dim = 768

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.vit(pixel_values)
        return outputs.last_hidden_state


class Decoder(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=2048, vocab_size=10000, feature_dim=768, num_layers=8, nhead=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.word2idx = None
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_feature = nn.Linear(feature_dim, embed_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def greedy_decode(self, features, max_len=MAX_SEQ_LEN):
        batch_size = features.size(0)
        
        memory = self.fc_feature(features)
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
    print(f"Device: {DEVICE}")
    print(f"Testing image: {image_path}")
    
    captions = load_captions(os.path.join(DATA_DIR, "caption.txt"))
    
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "final_model.pt"), map_location=DEVICE)
    word2idx = checkpoint["vocab"]
    vocab = Vocabulary(word2idx)
    print(f"Vocabulary size: {vocab.n_words}")
    
    model = CaptionModel(vocab.n_words, word2idx).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.decoder.word2idx = word2idx
    
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(DEVICE)
    
    img_name = os.path.basename(image_path)
    actual_caption = captions.get(img_name, "No caption available")
    
    print(f"Actual caption: {actual_caption}")
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            features = model.encoder(pixel_values)
            predictions = model.decoder.greedy_decode(features, MAX_SEQ_LEN)
    
    generated_caption = vocab.decode(predictions[0].cpu().numpy())
    
    print(f"Generated caption: {generated_caption}")
    
    if actual_caption != "No caption available":
        scores = calculate_metrics(actual_caption, generated_caption)
    else:
        scores = {'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0, 'meteor': 0, 'rouge_l': 0}
    
    output_path = os.path.join(OUTPUT_DIR, "sample_output.jpg")
    img.save(output_path)
    print(f"\nImage saved to: {output_path}")
    print(f"\n{'='*60}")
    print(f"Image: {img_name}")
    print(f"Generated: {generated_caption}")
    print(f"Actual:    {actual_caption}")
    print(f"{'-'*60}")
    print(f"Scores:")
    print(f"  BLEU-1:  {scores['bleu1']:.4f}")
    print(f"  BLEU-2:  {scores['bleu2']:.4f}")
    print(f"  BLEU-3:  {scores['bleu3']:.4f}")
    print(f"  BLEU-4:  {scores['bleu4']:.4f}")
    print(f"  METEOR:  {scores['meteor']:.4f}")
    print(f"  ROUGE-L: {scores['rouge_l']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(DATA_DIR, "Pictures", "6001.jpg")
    
    test_image(image_path)
