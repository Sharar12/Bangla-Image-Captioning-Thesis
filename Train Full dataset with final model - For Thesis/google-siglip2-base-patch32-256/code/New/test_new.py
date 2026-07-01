import os
import sys

# Silences PyTorch distributed elastic and extension warnings in the terminal
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["PYTORCH_MULTIPROCESSING_REDIRECTS"] = "0"
import logging

logging.getLogger("torch").setLevel(logging.ERROR)

import io
import random
import torch
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image, ImageTk
from collections import defaultdict
import tkinter as tk
from tqdm import tqdm
import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

# 👇 CLEANED: MODEL_DIR is completely gone. Only paths to your data and output exist.
DATA_DIR = r"D:\Python Projects\Train Full dataset with final model\Final_Hybrid_Dataset_60k"
OUTPUT_DIR = r"D:\Python Projects\Train Full dataset with final model - For Thesis\google-siglip2-base-patch32-256\output_new"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 32
BATCH_SIZE = 128

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
            w = self.idx2word.get(t, "<UNK>")
            if w in ["<EOS>", "<PAD>"]:
                break
            if w not in ["<SOS>", "<UNK>"]:
                words.append(w)
        return " ".join(words)


# 👇 CLEANED: The word 'siglip' has been completely removed from your local source definitions.
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 768

    def forward(self, pixel_values):
        # Dynamically grabs whatever internal vision modules PyTorch restored from your standalone .pt weights
        vision_attr = [v for k, v in self._modules.items() if "vision" in k or "siglip" in k]
        if vision_attr:
            return vision_attr[0].vision_model(pixel_values).last_hidden_state
        # Fallback direct access if unpickled as a flat layer structure
        return getattr(self, list(self._modules.keys())[0]).vision_model(pixel_values).last_hidden_state


class Decoder(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=2048, vocab_size=10000, feature_dim=768, num_layers=6, nhead=12,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.word2idx = None
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_feature = nn.Linear(feature_dim, embed_dim)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, features, captions):
        seq_len = captions.size(1)
        embed = self.embedding(captions) * (self.embed_dim ** 0.5)
        embed = embed + self.pos_encoder[:, :seq_len, :]
        memory = self.fc_feature(features)
        memory = self.encoder(memory)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        decoder_output = self.decoder(embed, memory, tgt_mask=tgt_mask)
        return self.fc_out(decoder_output)

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
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=vocab_size)

    def forward(self, pixel_values, captions):
        return self.decoder(self.encoder(pixel_values), captions)


def load_captions(caption_file):
    captions = defaultdict(list)
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                captions[parts[0]].append(parts[1])
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
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        all_caps = self.captions.get(img_name, [""])
        tokens = self.vocab.encode(random.choice(all_caps), MAX_SEQ_LEN)
        return pixel_values, torch.tensor(tokens), img_name


def calculate_metrics(reference_list, hypothesis):
    hypothesis_tokens = hypothesis.split()
    list_of_refs_tokens = [ref.split() for ref in reference_list]
    smoothing = SmoothingFunction().method4
    bleu1 = sentence_bleu(list_of_refs_tokens, hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    try:
        meteor = meteor_score(list_of_refs_tokens, hypothesis_tokens)
    except:
        meteor = 0.0
    rouge_scores = [calculate_rouge_l(ref.split(), hypothesis_tokens) for ref in reference_list]
    return {'bleu1': bleu1, 'meteor': meteor, 'rouge_l': max(rouge_scores)}


def calculate_rouge_l(reference, hypothesis):
    ref_set, hyp_set = set(reference), set(hypothesis)
    if len(ref_set) == 0 or len(hyp_set) == 0: return 0.0
    overlap = len(ref_set.intersection(hyp_set))
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_cider(reference_list, hypothesis):
    def get_ngrams(tokens, n):
        return list(ngrams(tokens, n))

    def cosine_similarity(ref_ngrams, hyp_ngrams):
        ref_freq, hyp_freq = defaultdict(int), defaultdict(int)
        for ng in ref_ngrams: ref_freq[ng] += 1
        for ng in hyp_ngrams: hyp_freq[ng] += 1
        all_ngrams = set(ref_freq.keys()) | set(hyp_freq.keys())
        ref_vec = [ref_freq.get(ng, 0) for ng in all_ngrams]
        hyp_vec = [hyp_freq.get(ng, 0) for ng in all_ngrams]
        ref_norm = math.sqrt(sum(x ** 2 for x in ref_vec))
        hyp_norm = math.sqrt(sum(x ** 2 for x in hyp_vec))
        if ref_norm == 0 or hyp_norm == 0: return 0.0
        return sum(ref_vec[i] * hyp_vec[i] for i in range(len(all_ngrams))) / (ref_norm * hyp_norm)

    hyp_tokens = hypothesis.split()
    scores = []
    for ref_tokens in [ref.split() for ref in reference_list]:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens): continue
            score += cosine_similarity(get_ngrams(ref_tokens, n), get_ngrams(hyp_tokens, n))
        scores.append(score / 4.0)
    return max(scores) if scores else 0.0


def show_sample_popup(img_path, generated_caption, ref_list, scores, img_name):
    root = tk.Tk()
    root.title("Random Sample Result")
    root.configure(bg="#1e293b")

    img = Image.open(img_path).convert("RGB")
    img.thumbnail((500, 380), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    main_frame = tk.Frame(root, bg="#1e293b", padx=20, pady=20)
    main_frame.pack()

    img_label = tk.Label(main_frame, image=tk_img, bg="#1e293b")
    img_label.image = tk_img
    img_label.pack(side="left", padx=(0, 20))

    text_frame = tk.Frame(main_frame, bg="#1e293b")
    text_frame.pack(side="left", fill="y")

    tk.Label(text_frame, text=f"📷 {img_name}", font=("Arial", 12, "bold"), fg="#e2e8f0", bg="#1e293b").pack(anchor="w")
    tk.Label(text_frame, text="\n📝 Generated Caption:", font=("Arial", 11, "bold"), fg="#34d399", bg="#1e293b").pack(
        anchor="w")
    tk.Label(text_frame, text=generated_caption, wraplength=400, justify="left", fg="#f1f5f9", bg="#1e293b",
             font=("Arial", 11)).pack(anchor="w", pady=2)

    tk.Label(text_frame, text="\n📖 Ground Truth References:", font=("Arial", 11, "bold"), fg="#60a5fa",
             bg="#1e293b").pack(anchor="w")
    for i, ref in enumerate(ref_list, 1):
        tk.Label(text_frame, text=f"{i}. {ref}", wraplength=400, justify="left", fg="#94a3b8", bg="#1e293b").pack(
            anchor="w", pady=1)

    scores_text = f"\n📈 Metrics Matrix:\n  BLEU-1: {scores['bleu1']:.4f}   METEOR: {scores['meteor']:.4f}\n  ROUGE-L: {scores['rouge_l']:.4f}   CIDEr: {scores.get('cider', 0):.4f}"
    tk.Label(text_frame, text=scores_text, justify="left", fg="#c084fc", bg="#1e293b", font=("Consolas", 10)).pack(
        anchor="w", pady=5)

    tk.Button(root, text="Exit Window", command=root.destroy, font=("Arial", 10), bg="#334155", fg="white", padx=20,
              pady=5).pack(pady=(0, 15))
    root.mainloop()


def test():
    print(f"Target Device: {DEVICE}")
    test_list = load_image_list(os.path.join(DATA_DIR, "test.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))

    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_20.pt")
    print(f"Loading Standalone Model from Checkpoint: {checkpoint_path}")

    # 👇 MAGIC TRICK: Maps old pickle references directly to our new clean local classes in execution memory
    current_module = sys.modules[__name__]
    sys.modules['__main__'] = current_module

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    vocab = Vocabulary(checkpoint["vocab"])
    model = checkpoint["full_model"].to(DEVICE)
    model.eval()

    # 👇 REPLACED: Downloads only the processing logic from the web repository config, bypasses local folders completely
    print("Fetching image preprocessing schema context directly from the web hub...")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch32-256", trust_remote_code=True)

    test_dataset = BanglaCaptionDataset(test_list, captions, vocab, processor, os.path.join(DATA_DIR, "images"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\nRunning Model Inference Loop...")
    all_results = []

    with torch.inference_mode():
        for pixel_values, _, img_names in tqdm(test_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                features = model.encoder(pixel_values)
                predictions = model.decoder.greedy_decode(features, max_len=MAX_SEQ_LEN)

            preds_cpu = predictions.cpu().numpy()
            for i, img_name in enumerate(img_names):
                gen_cap = vocab.decode(preds_cpu[i])
                ref_list = captions.get(img_name, [])
                if ref_list:
                    all_results.append((img_name, ref_list, gen_cap))

    print("\nProcessing Score Evaluations...")
    all_metrics = {'bleu1': [], 'meteor': [], 'rouge_l': [], 'cider': []}
    detailed_results = []

    for img_name, ref_list, hyp in tqdm(all_results, desc="Metrics"):
        m = calculate_metrics(ref_list, hyp)
        m['cider'] = calculate_cider(ref_list, hyp)
        for k in all_metrics: all_metrics[k].append(m[k])
        detailed_results.append({'img_name': img_name, 'reference': ref_list, 'hypothesis': hyp, **m})

    print("\n" + "=" * 50 + "\nFINAL TRANSLATION SCORES\n" + "=" * 50)
    for metric, scores in all_metrics.items():
        print(f"{metric.upper():10s}: {sum(scores) / len(scores):.4f}")
    print("=" * 50)

    pd.DataFrame([{k: sum(v) / len(v) for k, v in all_metrics.items()}]).to_csv(
        os.path.join(OUTPUT_DIR, "test_results.csv"), index=False)
    pd.DataFrame(detailed_results).to_csv(os.path.join(OUTPUT_DIR, "detailed_results.csv"), index=False)

    random_sample = random.choice(detailed_results)
    img_path = os.path.join(DATA_DIR, "images", random_sample['img_name'])

    show_sample_popup(
        img_path=img_path,
        generated_caption=random_sample["hypothesis"],
        ref_list=random_sample["reference"],
        scores={k: random_sample[k] for k in ["bleu1", "meteor", "rouge_l", "cider"]},
        img_name=random_sample["img_name"]
    )


if __name__ == "__main__":
    test()