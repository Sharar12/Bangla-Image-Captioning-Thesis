import os
import sys
import io
import random
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

MODEL_NAME = "nvidia/C-RADIOv2-B"
HF_TOKEN = "YOUR_HF_TOKEN"

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 32
BATCH_SIZE = 128

FEATURE_DIM = 768
EMBED_DIM = 768
NUM_LAYERS = 10
HIDDEN_DIM = 4096
NHEAD = 12
DROPOUT = 0.3

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


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

    def encode(self, sentence, max_len):
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sentence.split()]
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
        if len(tokens) < max_len:
            tokens += [self.word2idx["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len - 1] + [self.word2idx["<EOS>"]]
        return tokens

    def decode(self, tokens):
        words = []
        for t in tokens:
            if isinstance(t, torch.Tensor):
                t = t.item()
            w = self.idx2word.get(t, "<UNK>")
            if w in ["<EOS>", "<PAD>"]:
                break
            words.append(w)
        return " ".join(words)


def load_captions(caption_file):
    captions = defaultdict(list)
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                img_name = parts[0]
                caption = parts[1]
                captions[img_name].append(caption)
    return captions


def load_image_list(list_file):
    image_list = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(": ")
            if len(parts) >= 2:
                image_list.append(parts[1])
            elif ".jpg" in line:
                image_list.append(line)
            else:
                image_list.append(parts[0])
    return image_list


class BanglaCaptionDataset(Dataset):
    def __init__(self, image_list, captions, vocab, processor, image_dir):
        self.image_list = image_list
        self.captions = captions
        self.vocab = vocab
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        all_caps = self.captions.get(img_name, [""])
        cap = random.choice(all_caps)
        tokens = self.vocab.encode(cap, MAX_SEQ_LEN)
        
        return pixel_values, torch.tensor(tokens), img_name


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.radio = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
        for param in self.radio.parameters():
            param.requires_grad = False
        self.feature_dim = FEATURE_DIM

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.radio(pixel_values)
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif hasattr(outputs, 'features'):
            return outputs.features
        elif hasattr(outputs, 'hidden_states'):
            return outputs.hidden_states[-1]
        else:
            return outputs[0] if isinstance(outputs, tuple) else outputs


class Decoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab_size=10000, feature_dim=FEATURE_DIM, num_layers=NUM_LAYERS, nhead=NHEAD, dropout=DROPOUT):
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

    def forward(self, features, captions):
        batch_size = captions.size(0)
        seq_len = captions.size(1)
        
        embed = self.embedding(captions) * (self.embed_dim ** 0.5)
        embed = embed + self.pos_encoder[:, :seq_len, :]
        
        memory = self.fc_feature(features)
        memory = self.encoder(memory)
        
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        
        decoder_output = self.decoder(embed, memory, tgt_mask=tgt_mask)
        
        out = self.fc_out(decoder_output)
        return out

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

    def forward(self, pixel_values, captions):
        features = self.encoder(pixel_values)
        outputs = self.decoder(features, captions)
        return outputs

    def greedy_decode(self, features, max_len=MAX_SEQ_LEN):
        return self.decoder.greedy_decode(features, max_len)


def calculate_metrics(reference_list, hypothesis):
    hypothesis = hypothesis.split()
    list_of_refs_tokens = [ref.split() for ref in reference_list]
    
    smoothing = SmoothingFunction().method1
    
    bleu1 = sentence_bleu(list_of_refs_tokens, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(list_of_refs_tokens, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu(list_of_refs_tokens, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(list_of_refs_tokens, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    try:
        meteor = meteor_score(list_of_refs_tokens, hypothesis)
    except:
        meteor = 0.0
    
    rouge_scores = [calculate_rouge_l(ref.split(), hypothesis) for ref in reference_list]
    rouge_l = max(rouge_scores)
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge_l': rouge_l
    }


def calculate_rouge_l(reference, hypothesis):
    ref_set = set(reference)
    hyp_set = set(hypothesis)
    
    if len(ref_set) == 0 or len(hyp_set) == 0:
        return 0.0
    
    intersection = len(ref_set & hyp_set)
    
    if intersection == 0:
        return 0.0
    
    precision = intersection / len(hyp_set)
    recall = intersection / len(ref_set)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print("Loading data...")
    
    test_list = load_image_list(os.path.join(DATA_DIR, "test.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "caption.txt"))
    
    print(f"Test images: {len(test_list)}")
    print(f"Captioned images: {len(captions)}")
    
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, "final_model.pt"), map_location=DEVICE)
    word2idx = checkpoint.get("vocab", None)
    vocab = Vocabulary(word2idx)
    print(f"Vocabulary size: {vocab.n_words}")
    
    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
    
    test_dataset = BanglaCaptionDataset(test_list, captions, vocab, processor, os.path.join(DATA_DIR, "Pictures"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = CaptionModel(vocab.n_words, word2idx).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("\nEvaluating on GPU...")
    
    all_results = []
    
    with torch.no_grad():
        for pixel_values, tokens, img_names in tqdm(test_loader, desc="Testing"):
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                features = model.encoder(pixel_values)
                predictions = model.decoder.greedy_decode(features, MAX_SEQ_LEN)
            
            preds_cpu = predictions.cpu().numpy()
            for i, img_name in enumerate(img_names):
                generated_caption = vocab.decode(preds_cpu[i])
                ref_list = captions.get(img_name, [])
                if ref_list:
                    all_results.append((img_name, ref_list, generated_caption))
    
    print("\nCalculating Metrics on CPU...")
    all_metrics = {
        'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': [],
        'meteor': [], 'rouge_l': []
    }
    detailed_results = []
    
    for img_name, ref_list, hyp in tqdm(all_results, desc="Metrics"):
        if ref_list:
            m = calculate_metrics(ref_list, hyp)
            for k in all_metrics:
                all_metrics[k].append(m[k])
            detailed_results.append({
                'img_name': img_name,
                'reference': ref_list,
                'hypothesis': hyp,
                'bleu1': m['bleu1'],
                'bleu2': m['bleu2'],
                'bleu3': m['bleu3'],
                'bleu4': m['bleu4'],
                'meteor': m['meteor'],
                'rouge_l': m['rouge_l']
            })
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for metric, scores in all_metrics.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{metric.upper():10s}: {avg:.4f}")
    
    print("=" * 60)
    print(f"Total samples evaluated: {len(all_metrics['bleu1'])}")
    print("=" * 60)
    
    results_summary = {}
    for metric, scores in all_metrics.items():
        results_summary[metric] = sum(scores) / len(scores) if scores else 0
    
    results_summary['samples'] = len(all_metrics['bleu1'])
    results_df = pd.DataFrame([results_summary])
    results_file = os.path.join(OUTPUT_DIR, "test_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    detailed_file = os.path.join(OUTPUT_DIR, "detailed_results.csv")
    pd.DataFrame(detailed_results).to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    random_idx = random.randint(0, len(detailed_results) - 1)
    sample = detailed_results[random_idx]
    
    img_path = os.path.join(DATA_DIR, "Pictures", sample['img_name'])
    img = Image.open(img_path).convert("RGB")
    img.save(os.path.join(OUTPUT_DIR, "random_sample.jpg"))
    
    print(f"\n{'='*70}")
    print(f"RANDOM SAMPLE (Index: {random_idx})")
    print(f"{'='*70}")
    print(f"Image: {sample['img_name']}")
    print(f"Generated: {sample['hypothesis']}")
    print(f"Actual:    {sample['reference']}")
    print(f"{'-'*70}")
    print(f"Scores:")
    print(f"  BLEU-1:  {sample['bleu1']:.4f}")
    print(f"  BLEU-2:  {sample['bleu2']:.4f}")
    print(f"  BLEU-3:  {sample['bleu3']:.4f}")
    print(f"  BLEU-4:  {sample['bleu4']:.4f}")
    print(f"  METEOR:  {sample['meteor']:.4f}")
    print(f"  ROUGE-L: {sample['rouge_l']:.4f}")
    print(f"{'='*70}")
    print(f"Image saved to: {os.path.join(OUTPUT_DIR, 'random_sample.jpg')}")


if __name__ == "__main__":
    test()
