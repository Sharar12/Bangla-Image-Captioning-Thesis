import os
import sys
import io
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import nltk
import math
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk import ngrams

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Download required NLTK data quietly
for resource in ['wordnet', 'punkt', 'punkt_tab']:
    nltk.download(resource, quiet=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(BASE_DIR)
DATA_DIR = r"D:\Python Projects\Train Full dataset with final model\Final_Hybrid_Dataset_60k"
MODEL_DIR = os.path.join(MODEL_FOLDER, "model")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        for param in self.siglip.parameters():
            param.requires_grad = False
        self.feature_dim = 768

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.siglip.vision_model(pixel_values)
        return outputs.last_hidden_state


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
        return self.decoder(features, captions)

    def greedy_decode(self, features, max_len=MAX_SEQ_LEN):
        return self.decoder.greedy_decode(features, max_len)


def load_captions(caption_file):
    captions = defaultdict(list)
    if not os.path.exists(caption_file):
        return captions
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                captions[parts[0]].append(parts[1])
    return captions


def calculate_rouge_l(reference, hypothesis):
    ref_set = set(reference)
    hyp_set = set(hypothesis)
    if len(ref_set) == 0 or len(hyp_set) == 0: return 0.0
    overlap = len(ref_set.intersection(hyp_set))
    precision = overlap / len(hyp_set) if len(hyp_set) > 0 else 0
    recall = overlap / len(ref_set) if len(ref_set) > 0 else 0
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(reference_list, hypothesis):
    hyp_tokens = hypothesis.split()
    refs_tokens = [ref.split() for ref in reference_list]
    smoothing = SmoothingFunction().method4
    bleu1 = sentence_bleu(refs_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    try:
        meteor = meteor_score(refs_tokens, hyp_tokens)
    except:
        meteor = 0.0
    rouge_l = max(calculate_rouge_l(ref.split(), hyp_tokens) for ref in reference_list)
    return {'bleu1': bleu1, 'meteor': meteor, 'rouge_l': rouge_l}


def calculate_cider(reference_list, hypothesis):
    def get_ngrams(tokens, n):
        return list(ngrams(tokens, n))

    def cosine_similarity(ref_ngrams, hyp_ngrams):
        ref_freq = defaultdict(int)
        hyp_freq = defaultdict(int)
        for ng in ref_ngrams: ref_freq[ng] += 1
        for ng in hyp_ngrams: hyp_freq[ng] += 1
        all_ng = set(ref_freq.keys()) | set(hyp_freq.keys())
        ref_vec = [ref_freq.get(ng, 0) for ng in all_ng]
        hyp_vec = [hyp_freq.get(ng, 0) for ng in all_ng]
        ref_norm = math.sqrt(sum(x ** 2 for x in ref_vec))
        hyp_norm = math.sqrt(sum(x ** 2 for x in hyp_vec))
        if ref_norm == 0 or hyp_norm == 0: return 0.0
        dot = sum(r * h for r, h in zip(ref_vec, hyp_vec))
        return dot / (ref_norm * hyp_norm)

    hyp_tokens = hypothesis.split()
    scores = []
    for ref in reference_list:
        ref_tokens = ref.split()
        score = 0.0
        for n in range(1, 5):
            if n > len(hyp_tokens) or n > len(ref_tokens): continue
            sim = cosine_similarity(get_ngrams(hyp_tokens, n), get_ngrams(ref_tokens, n))
            score += sim
        scores.append(score / 4.0)
    return max(scores) if scores else 0.0


def select_image():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Image for Captioning",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
    )
    root.destroy()
    return path


def show_results_gui(image_path, generated_caption, ref_list, metrics):
    root = tk.Tk()
    root.title("Image Captioning Result")
    root.resizable(False, False)

    # Load & resize image
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((550, 400), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        img_label = tk.Label(root, image=tk_img)
        img_label.image = tk_img  # Prevent garbage collection
        img_label.pack(pady=10)
    except Exception as e:
        tk.Label(root, text=f"⚠️ Could not load image: {e}", fg="red").pack(pady=10)

    # Content frame
    content = tk.Frame(root, padx=20, pady=10)
    content.pack(fill="both", expand=True)

    # Generated caption
    tk.Label(content, text="📝 Generated Caption:", font=("Arial", 11, "bold")).pack(anchor="w")
    tk.Label(content, text=generated_caption, wraplength=540, justify="left", fg="blue").pack(anchor="w", pady=5)

    # References & Metrics
    if ref_list:
        tk.Label(content, text="📖 Reference(s):", font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 0))
        for i, ref in enumerate(ref_list, 1):
            tk.Label(content, text=f"{i}. {ref}", wraplength=540, justify="left").pack(anchor="w", pady=2)

        if metrics:
            tk.Label(content, text="📈 Metrics:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 0))
            metrics_str = (
                f"BLEU-1 : {metrics['bleu1']:.4f}\n"
                f"METEOR : {metrics['meteor']:.4f}\n"
                f"ROUGE-L: {metrics['rouge_l']:.4f}\n"
                f"CIDER  : {metrics['cider']:.4f}"
            )
            tk.Label(content, text=metrics_str, justify="left", font=("Consolas", 10)).pack(anchor="w", pady=5)
    else:
        tk.Label(content, text="⚠️ No reference captions available for metrics.", fg="orange").pack(anchor="w", pady=10)

    tk.Button(root, text="Close Window", command=root.destroy, font=("Arial", 10)).pack(pady=10)
    root.mainloop()


def test_single_image():
    print("📂 Opening file dialog...")
    test_image_path = select_image()
    if not test_image_path:
        print("⚠️ No image selected. Exiting.")
        return

    img_name = os.path.basename(test_image_path)
    print(f"✅ Selected: {img_name}")
    print(f"💻 Device: {DEVICE}")
    print("📦 Loading model & vocabulary...")

    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_20.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    word2idx = checkpoint["vocab"]
    vocab = Vocabulary(word2idx)

    model = CaptionModel(vocab.n_words, word2idx).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("✨ Model loaded successfully.")

    print("🖼️  Processing image...")
    image = Image.open(test_image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            features = model.encoder(pixel_values)
            predictions = model.greedy_decode(features, max_len=MAX_SEQ_LEN)

    generated_caption = vocab.decode(predictions[0].cpu().numpy().tolist())

    # Find references
    ref_list = []
    caption_file = os.path.join(DATA_DIR, "captions.txt")
    if os.path.exists(caption_file):
        all_caps = load_captions(caption_file)
        ref_list = all_caps.get(img_name, [])

    if ref_list:
        print(f"📖 Found {len(ref_list)} reference caption(s) in dataset.")
    else:
        print("⚠️ Image not found in captions.txt.")
        manual_ref = input("📝 Type a reference caption to calculate metrics (or press Enter to skip): ").strip()
        if manual_ref:
            ref_list = [manual_ref]

    # Calculate metrics
    metrics = {}
    if ref_list:
        metrics = calculate_metrics(ref_list, generated_caption)
        metrics['cider'] = calculate_cider(ref_list, generated_caption)
        print("\n📈 METRICS SCORES")
        for k, v in metrics.items():
            print(f"  {k.upper():7s}: {v:.4f}")
    else:
        print("\n⚠️ Skipping metrics (no reference captions provided).")

    # Show GUI (keeps window open until closed)
    show_results_gui(test_image_path, generated_caption, ref_list, metrics if metrics else None)
    print("\n✅ GUI closed. Script finished.")


if __name__ == "__main__":
    test_single_image()