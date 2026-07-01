import os
import sys
import gc
import math
import warnings
import logging
import io
from collections import defaultdict
import tkinter as tk
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageTk
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

TARGET_IMAGE = "1.jpg"

warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils._pytree")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["PYTORCH_MULTIPROCESSING_REDIRECTS"] = "0"
logging.getLogger("torch").setLevel(logging.ERROR)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = r"D:\Python Projects\Research Papers Test\BNNATURE"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total_vram - 1e9) / total_vram)
    torch.cuda.empty_cache()

MAX_SEQ_LEN = 51
BATCH_SIZE = 64
FEATURE_DIM = 2048
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.1

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


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
            if w != "<SOS>":
                words.append(w)
        return " ".join(words)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(weights="DEFAULT")
        self.inception.fc = nn.Identity()
        self.inception.AuxLogits = nn.Identity()
        self.inception.avgpool = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False
        self.feature_dim = FEATURE_DIM
        self.inception.Mixed_7c.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        self._features = output

    def forward(self, pixel_values):
        with torch.no_grad():
            self.inception(pixel_values)
        x = self._features
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)
        return x


class BanglaLuongAttentionDecoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.word2idx = None
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attn_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, input_ids, hidden=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        embeddings = self.dropout(self.embedding(input_ids))
        if hidden is None:
            hidden = torch.zeros(1, batch_size, HIDDEN_DIM, device=device)
        gru_out, hidden = self.gru(embeddings, hidden)
        adapted = self.attn_W(encoder_outputs)
        scores = torch.bmm(gru_out, adapted.transpose(1, 2))
        alignment = torch.softmax(scores, dim=-1)
        context = torch.bmm(alignment, encoder_outputs)
        combined = torch.cat([context, gru_out], dim=-1)
        fused = self.tanh(self.concat(combined))
        logits = self.proj(self.dropout(fused))
        return logits

    def step(self, token_embed, hidden, encoder_outputs):
        gru_out, hidden = self.gru(token_embed, hidden)
        adapted = self.attn_W(encoder_outputs)
        scores = torch.bmm(gru_out, adapted.transpose(1, 2))
        alignment = torch.softmax(scores, dim=-1)
        context = torch.bmm(alignment, encoder_outputs)
        combined = torch.cat([context, gru_out], dim=-1)
        fused = self.tanh(self.concat(combined))
        logits = self.proj(fused)
        return logits, hidden

    @torch.no_grad()
    def greedy_decode(self, encoder_outputs, max_len=MAX_SEQ_LEN):
        batch_size = encoder_outputs.size(0)
        hidden = None
        generated = torch.full(
            (batch_size, 1),
            self.word2idx["<SOS>"],
            dtype=torch.long,
        ).to(encoder_outputs.device)
        for _ in range(max_len - 1):
            embed = self.embedding(generated[:, -1:])
            logits, hidden = self.step(embed, hidden, encoder_outputs)
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.word2idx["<EOS>"]).all():
                break
        return generated


class CaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = BanglaLuongAttentionDecoder(vocab_size=vocab_size)

    def forward(self, pixel_values, captions):
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        return self.decoder(visual_tokens, captions)

    @torch.no_grad()
    def generate_captions(self, pixel_values, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        predictions = self.decoder.greedy_decode(visual_tokens, max_new_tokens)
        return predictions

    @torch.no_grad()
    def generate_caption(self, pixel_values, max_new_tokens=MAX_SEQ_LEN):
        return self.generate_captions(pixel_values, max_new_tokens)[0]


def load_captions(caption_file):
    captions = {}
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                fname = parts[0]
                cap = parts[1]
                if fname not in captions:
                    captions[fname] = []
                captions[fname].append(cap)
    return captions


def load_image_list(list_file):
    image_list = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_list.append(line)
    return image_list


class BanglaCaptionDataset(Dataset):
    def __init__(self, image_list, captions, vocab, image_dir):
        self.image_list = image_list
        self.captions = captions
        self.vocab = vocab
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        pixel_values = self.transform(image)
        all_caps = self.captions.get(img_name, [""])
        return pixel_values, all_caps, img_name


def calculate_metrics(reference_list, hypothesis):
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    list_of_refs_tokens = [nltk.word_tokenize(ref) for ref in reference_list]
    smoothing = SmoothingFunction().method4

    bleu1 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing,
    )
    bleu2 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing,
    )
    bleu3 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothing,
    )
    bleu4 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    try:
        meteor = meteor_score(list_of_refs_tokens, hypothesis_tokens)
    except Exception:
        meteor = 0.0

    rouge_scores = [
        calculate_rouge_l(nltk.word_tokenize(ref), hypothesis_tokens)
        for ref in reference_list
    ]
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": max(rouge_scores) if rouge_scores else 0.0,
    }


def calculate_rouge_l(reference, hypothesis):
    ref_set, hyp_set = set(reference), set(hypothesis)
    if len(ref_set) == 0 or len(hyp_set) == 0:
        return 0.0
    overlap = len(ref_set.intersection(hyp_set))
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return (
        (2 * (precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )


def calculate_cider(reference_list, hypothesis):
    def get_ngrams(tokens, n):
        return list(ngrams(tokens, n))

    def cosine_similarity(ref_ngrams, hyp_ngrams):
        ref_freq, hyp_freq = defaultdict(int), defaultdict(int)
        for ng in ref_ngrams:
            ref_freq[ng] += 1
        for ng in hyp_ngrams:
            hyp_freq[ng] += 1
        all_ngrams = set(ref_freq.keys()) | set(hyp_freq.keys())
        ref_vec = [ref_freq.get(ng, 0) for ng in all_ngrams]
        hyp_vec = [hyp_freq.get(ng, 0) for ng in all_ngrams]
        ref_norm = math.sqrt(sum(x**2 for x in ref_vec))
        hyp_norm = math.sqrt(sum(x**2 for x in hyp_vec))
        if ref_norm == 0 or hyp_norm == 0:
            return 0.0
        return sum(ref_vec[i] * hyp_vec[i] for i in range(len(all_ngrams))) / (
            ref_norm * hyp_norm
        )

    hyp_tokens = nltk.word_tokenize(hypothesis)
    scores = []
    for ref_tokens in [nltk.word_tokenize(ref) for ref in reference_list]:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens):
                continue
            score += cosine_similarity(
                get_ngrams(ref_tokens, n), get_ngrams(hyp_tokens, n)
            )
        scores.append(score / 4.0)
    return max(scores) if scores else 0.0


def show_initial_popup(img_path, target_img, target_cap, target_refs, target_scores):
    root = tk.Tk()
    root.title("Target Image Preview Dashboard")
    root.configure(bg="#1e293b")

    img = Image.open(img_path).convert("RGB")
    img.thumbnail((450, 340), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    main_frame = tk.Frame(root, bg="#1e293b", padx=20, pady=20)
    main_frame.pack()

    img_label = tk.Label(main_frame, image=tk_img, bg="#1e293b")
    img_label.image = tk_img
    img_label.pack(side="left", padx=(0, 20), anchor="n")

    text_frame = tk.Frame(main_frame, bg="#1e293b")
    text_frame.pack(side="left", fill="both", expand=True)

    tk.Label(
        text_frame,
        text=f"Analyzed Target Image: {target_img}",
        font=("Arial", 12, "bold"),
        fg="#e2e8f0",
        bg="#1e293b",
    ).pack(anchor="w")
    tk.Label(
        text_frame,
        text="Generated Caption (Processed):",
        font=("Arial", 10, "bold"),
        fg="#34d399",
        bg="#1e293b",
    ).pack(anchor="w", pady=(5, 0))
    tk.Label(
        text_frame,
        text=target_cap,
        wraplength=450,
        justify="left",
        fg="#f1f5f9",
        bg="#1e293b",
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)

    tk.Label(
        text_frame,
        text="\nIsolated Image Score Metrics Matrix:",
        font=("Arial", 11, "bold"),
        fg="#38bdf8",
        bg="#1e293b",
    ).pack(anchor="w", pady=(5, 5))

    matrix_frame = tk.Frame(text_frame, bg="#0f172a", bd=1, relief="solid")
    matrix_frame.pack(anchor="w", fill="x", expand=True)

    tk.Label(
        matrix_frame,
        text="Metric",
        font=("Consolas", 10, "bold"),
        fg="#94a3b8",
        bg="#0f172a",
        width=15,
        anchor="w",
    ).grid(row=0, column=0, padx=10, pady=4)
    tk.Label(
        matrix_frame,
        text="Score Value",
        font=("Consolas", 10, "bold"),
        fg="#a78bfa",
        bg="#0f172a",
        width=18,
        anchor="e",
    ).grid(row=0, column=1, padx=10, pady=4)

    metrics_list = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]
    for idx, m_key in enumerate(metrics_list, start=1):
        bg_color = "#1e293b" if idx % 2 == 0 else "#0f172a"
        tk.Label(
            matrix_frame,
            text=m_key.upper(),
            font=("Consolas", 10),
            fg="#cbd5e1",
            bg=bg_color,
            width=15,
            anchor="w",
        ).grid(row=idx, column=0, padx=10, pady=2)
        tk.Label(
            matrix_frame,
            text=f"{target_scores[m_key]:.4f}",
            font=("Consolas", 10),
            fg="#c084fc",
            bg=bg_color,
            width=18,
            anchor="e",
        ).grid(row=idx, column=1, padx=10, pady=2)

    tk.Button(
        root,
        text="Start Full Average Evaluation",
        command=root.destroy,
        font=("Arial", 10, "bold"),
        bg="#10b981",
        fg="white",
        padx=25,
        pady=6,
    ).pack(pady=15)
    root.mainloop()


def run_evaluation():
    print(f"Target Device: {DEVICE}")

    test_txt = os.path.join(DATA_DIR, "caption", "test.txt")
    if not os.path.exists(test_txt):
        print("Error: test.txt not found!")
        return

    test_list = load_image_list(test_txt)
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))

    checkpoint_path = os.path.join(OUTPUT_DIR, "final_model_4bit.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {OUTPUT_DIR}")
        return
    print(f"Loading weights from checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    vocab_size = checkpoint.get(
        "vocab_size", checkpoint["model_state_dict"]["decoder.embedding.weight"].size(0)
    )

    model = CaptionModel(vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.decoder.word2idx = checkpoint.get("word2idx", None)
    if model.decoder.word2idx is None:
        print(
            "Warning: word2idx not found in checkpoint, generating vocab from captions..."
        )
        vocab = Vocabulary()
        for caps in captions.values():
            for cap in caps:
                for word in cap.split():
                    if word not in vocab.word2idx:
                        idx = len(vocab.word2idx)
                        vocab.word2idx[word] = idx
                        vocab.idx2word[idx] = word
        model.decoder.word2idx = vocab.word2idx

    model = model.to(DEVICE)
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    vocab_obj = Vocabulary(model.decoder.word2idx)

    # =========================================================================
    # PHASE 1: IMMEDIATE TARGET IMAGE INFERENCE
    # =========================================================================
    print(f"\n[PHASE 1] Instantly evaluating isolated target image: {TARGET_IMAGE}...")
    target_img_path = os.path.join(DATA_DIR, "images", TARGET_IMAGE)
    target_captured_data = None

    if os.path.exists(target_img_path):
        ref_list = captions.get(TARGET_IMAGE, [])
        if ref_list:
            transform = transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            image = Image.open(target_img_path).convert("RGB")
            pixel_values = transform(image).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                predictions = model.generate_captions(pixel_values)
                gen_cap = vocab_obj.decode(predictions[0].cpu().numpy().tolist())

            m = calculate_metrics(ref_list, gen_cap)
            m["cider"] = calculate_cider(ref_list, gen_cap)

            target_captured_data = {
                "img_name": TARGET_IMAGE,
                "gen_caption": gen_cap,
                "references": ref_list,
                "scores": m,
                "path": target_img_path,
            }

            print("=" * 65)
            print(
                f"ISOLATED TARGET IMAGE EVALUATION: {target_captured_data['img_name']}"
            )
            print("=" * 65)
            print(f"Generated : {target_captured_data['gen_caption']}")
            print(f"References: {target_captured_data['references']}\n")
            print(f"{'METRIC':12s} | {'SCORE':14s}")
            print("-" * 65)
            for k in ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]:
                print(f"{k.upper():12s} | {target_captured_data['scores'][k]:14.4f}")
            print("=" * 65)

            show_initial_popup(
                img_path=target_captured_data["path"],
                target_img=target_captured_data["img_name"],
                target_cap=target_captured_data["gen_caption"],
                target_refs=target_captured_data["references"],
                target_scores=target_captured_data["scores"],
            )
        else:
            print(
                f"Warning: Reference ground truths missing for {TARGET_IMAGE} inside caption.txt"
            )
    else:
        print(f"Warning: Target image path not found at {target_img_path}")

    # =========================================================================
    # PHASE 2: GLOBAL FULL AVERAGES
    # =========================================================================
    print(
        f"\n[PHASE 2] Starting full evaluation partition over {len(test_list)} images..."
    )
    all_metrics = {
        "bleu1": [],
        "bleu2": [],
        "bleu3": [],
        "bleu4": [],
        "meteor": [],
        "rouge_l": [],
        "cider": [],
    }

    for img_name in tqdm(test_list, desc="Processing Entire Test Dataset"):
        img_path = os.path.join(DATA_DIR, "images", img_name)
        if not os.path.exists(img_path):
            continue

        if target_captured_data and img_name.lower() == TARGET_IMAGE.lower():
            for k in all_metrics:
                all_metrics[k].append(target_captured_data["scores"][k])
            continue

        transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(img_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            predictions = model.generate_captions(pixel_values)
            gen_cap = vocab_obj.decode(predictions[0].cpu().numpy().tolist())

        ref_list = captions.get(img_name, [])
        if not ref_list:
            continue

        m = calculate_metrics(ref_list, gen_cap)
        m["cider"] = calculate_cider(ref_list, gen_cap)

        for k in all_metrics:
            all_metrics[k].append(m[k])

    dataset_averages = {
        k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()
    }

    print("\n" + "=" * 65)
    print(f"FINAL OVERALL DATASET AVERAGE SCORES ({len(test_list)} Images)")
    print("=" * 65)
    for metric, score in dataset_averages.items():
        print(f"{metric.upper():10s}: {score:.4f}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame([dataset_averages]).to_csv(
        os.path.join(OUTPUT_DIR, "test_overall_averages.csv"), index=False
    )
    print(f"Process completely executed successfully. Records cached at {OUTPUT_DIR}")


if __name__ == "__main__":
    run_evaluation()
