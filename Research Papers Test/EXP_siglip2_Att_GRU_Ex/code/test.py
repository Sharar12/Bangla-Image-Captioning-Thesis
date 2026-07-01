import os
import json
import re
import sys
import gc
import math
import warnings
import logging
import io
import unicodedata
from collections import defaultdict
import tkinter as tk
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image, ImageTk
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

TARGET_IMAGE = "1.jpg"

warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["PYTORCH_MULTIPROCESSING_REDIRECTS"] = "0"
logging.getLogger("torch").setLevel(logging.ERROR)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = r"D:\Python Projects\Research Papers Test\BNNATURE"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_PATH = os.path.join(MODELS_DIR, "BanglaGPT")
os.environ["HF_HOME"] = os.path.join(MODEL_FOLDER, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total_vram - 1e9) / total_vram)
    torch.cuda.empty_cache()

MAX_SEQ_LEN = 51
BATCH_SIZE = 64
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.1
GRU_NUM_LAYERS = 8
LSTM_NUM_LAYERS = 8
BANGLA_GPT_HIDDEN = 768

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.count = 4

    def decode(self, ids):
        words = []
        for idx in ids:
            w = self.idx2word.get(idx, "<unk>")
            if w == "<end>":
                break
            if w not in ("<pad>", "<start>"):
                words.append(w)
        return " ".join(words)

    @property
    def vocab_size(self):
        return self.count

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v.word2idx = data["word2idx"]
        v.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        v.count = len(v.word2idx)
        return v


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_output, encoder_outputs):
        adapted = self.attn_W(encoder_outputs)
        scores = torch.bmm(decoder_output, adapted.transpose(1, 2))
        alignment = torch.softmax(scores, dim=-1)
        return torch.bmm(alignment, encoder_outputs)


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, num_layers=GRU_NUM_LAYERS, batch_first=True
        )
        self.attention = AttentionLayer(hidden_dim)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, encoder_outputs, input_ids, hidden=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        embeddings = self.dropout(self.embedding(input_ids))
        if hidden is None:
            hidden = torch.zeros(GRU_NUM_LAYERS, batch_size, HIDDEN_DIM, device=device)
        rnn_outputs, hidden = self.rnn(embeddings, hidden)
        context = self.attention(rnn_outputs, encoder_outputs)
        combined = torch.cat([context, rnn_outputs], dim=-1)
        fused = self.tanh(self.concat(combined))
        return self.proj(self.dropout(fused))

    def step(self, token_embed, hidden, encoder_outputs):
        rnn_output, hidden = self.rnn(token_embed, hidden)
        context = self.attention(rnn_output, encoder_outputs)
        combined = torch.cat([context, rnn_output], dim=-1)
        fused = self.tanh(self.concat(combined))
        return self.proj(fused), hidden


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=LSTM_NUM_LAYERS, batch_first=True
        )
        self.attention = AttentionLayer(hidden_dim)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, encoder_outputs, input_ids, hidden=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        embeddings = self.dropout(self.embedding(input_ids))
        if hidden is None:
            h0 = torch.zeros(LSTM_NUM_LAYERS, batch_size, HIDDEN_DIM, device=device)
            c0 = torch.zeros(LSTM_NUM_LAYERS, batch_size, HIDDEN_DIM, device=device)
            hidden = (h0, c0)
        rnn_outputs, hidden = self.rnn(embeddings, hidden)
        context = self.attention(rnn_outputs, encoder_outputs)
        combined = torch.cat([context, rnn_outputs], dim=-1)
        fused = self.tanh(self.concat(combined))
        return self.proj(self.dropout(fused))

    def step(self, token_embed, hidden, encoder_outputs):
        rnn_output, hidden = self.rnn(token_embed, hidden)
        context = self.attention(rnn_output, encoder_outputs)
        combined = torch.cat([context, rnn_output], dim=-1)
        fused = self.tanh(self.concat(combined))
        return self.proj(fused), hidden


class BanglaGPTDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.vis_proj = nn.Linear(HIDDEN_DIM, BANGLA_GPT_HIDDEN)
        self.gpt_proj = nn.Linear(EMBED_DIM, BANGLA_GPT_HIDDEN)
        model_path = (
            BANGLA_GPT_PATH
            if os.path.exists(BANGLA_GPT_PATH)
            else "shahidul034/BanglaGPT"
        )
        self.gpt = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.output_proj = nn.Linear(BANGLA_GPT_HIDDEN, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, visual_tokens, input_ids):
        vis = self.vis_proj(visual_tokens)
        text_embeds = self.gpt_proj(self.embedding(input_ids))
        combined = torch.cat([vis, text_embeds], dim=1)
        gpt_out = self.gpt(inputs_embeds=combined).last_hidden_state
        return self.output_proj(self.dropout(gpt_out[:, -input_ids.size(1) :]))


class MultiDecoderEnsemble(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.gru = GRUDecoder(vocab_size)
        self.lstm = LSTMDecoder(vocab_size)
        self.banglagpt = BanglaGPTDecoder(vocab_size)

    def forward(self, encoder_outputs, input_ids):
        logits_gru = self.gru(encoder_outputs, input_ids)
        logits_lstm = self.lstm(encoder_outputs, input_ids)
        logits_gpt = self.banglagpt(encoder_outputs, input_ids)
        return (logits_gru + logits_lstm + logits_gpt) / 3.0


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_path = (
            SIGLIP_MODEL_PATH
            if os.path.exists(SIGLIP_MODEL_PATH)
            else "google/siglip2-base-patch32-256"
        )
        self.siglip = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        for param in self.siglip.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        return self.siglip.vision_model(pixel_values).last_hidden_state


class CaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = MultiDecoderEnsemble(vocab_size)

    def forward(self, pixel_values, input_ids):
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        return self.decoder(visual_tokens, input_ids)

    @torch.no_grad()
    def generate_captions(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        start_id, end_id = vocab.word2idx["<start>"], vocab.word2idx["<end>"]
        generated = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=device
        )
        gru_hidden = None
        lstm_hidden = None

        for _ in range(max_new_tokens - 1):
            logits_list = []

            token_gru = self.decoder.gru.embedding.weight[generated[:, -1:]]
            logits_g, gru_hidden = self.decoder.gru.step(
                token_gru, gru_hidden, visual_tokens
            )
            logits_list.append(logits_g)

            token_lstm = self.decoder.lstm.embedding.weight[generated[:, -1:]]
            logits_l, lstm_hidden = self.decoder.lstm.step(
                token_lstm, lstm_hidden, visual_tokens
            )
            logits_list.append(logits_l)

            vis_gpt = self.decoder.banglagpt.vis_proj(visual_tokens)
            text_embeds_gpt = self.decoder.banglagpt.gpt_proj(
                self.decoder.banglagpt.embedding(generated)
            )
            comb_gpt = torch.cat([vis_gpt, text_embeds_gpt], dim=1)
            gpt_out = self.decoder.banglagpt.gpt(
                inputs_embeds=comb_gpt
            ).last_hidden_state
            logits_b = self.decoder.banglagpt.output_proj(gpt_out[:, -1:])
            logits_list.append(logits_b)

            avg_logits = sum(logits_list) / len(logits_list)
            next_token = avg_logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return [vocab.decode(seq.tolist()) for seq in generated]

    def generate_caption(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        return self.generate_captions(pixel_values, vocab, max_new_tokens)[0]


def load_captions(caption_file):
    captions = {}
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                fname, cap = parts[0], parts[1]
                if fname not in captions:
                    captions[fname] = []
                captions[fname].append(cap)
    return captions


def load_image_list(list_file):
    return [line.strip() for line in open(list_file, "r") if line.strip()]


class BanglaCaptionDataset(Dataset):
    def __init__(self, image_list, captions, processor, image_dir):
        self.image_list = image_list
        self.captions = captions
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
    if not ref_set or not hyp_set:
        return 0.0
    overlap = len(ref_set & hyp_set)
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return (
        (2 * precision * recall / (precision + recall))
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

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.count} tokens")

    test_txt = os.path.join(DATA_DIR, "caption", "test.txt")
    if not os.path.exists(test_txt):
        print("Error: test.txt not found!")
        return

    test_list = load_image_list(test_txt)
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))

    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        return
    print(f"Loading best model: {best_model_path}")

    current_module = sys.modules[__name__]
    sys.modules["__main__"] = current_module
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)

    print("Building model (SigLIP2 → MultiDecoder Ensemble)...")
    model = CaptionModel(vocab_size=vocab.vocab_size).to(DEVICE)
    print("Loading trained weights...")
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    model.eval()

    print("Loading SigLIP2 image processor...")
    processor_path = (
        SIGLIP_MODEL_PATH
        if os.path.exists(os.path.join(SIGLIP_MODEL_PATH, "preprocessor_config.json"))
        else "google/siglip2-base-patch32-256"
    )
    processor = AutoImageProcessor.from_pretrained(
        processor_path, trust_remote_code=True
    )

    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 1: IMMEDIATE TARGET IMAGE INFERENCE
    # =========================================================================
    print(f"\n[PHASE 1] Instantly evaluating isolated target image: {TARGET_IMAGE}...")
    target_img_path = os.path.join(DATA_DIR, "images", TARGET_IMAGE)
    target_captured_data = None

    if os.path.exists(target_img_path):
        ref_list = captions.get(TARGET_IMAGE, [])
        if ref_list:
            image = Image.open(target_img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].to(DEVICE)

            with torch.inference_mode():
                gen_cap = model.generate_caption(pixel_values, vocab)
                if "।" in gen_cap:
                    gen_cap = gen_cap.split("।")[0] + "।"

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

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(
            DEVICE
        )

        with torch.inference_mode():
            gen_cap = model.generate_caption(pixel_values, vocab)
            if "।" in gen_cap:
                gen_cap = gen_cap.split("।")[0] + "।"

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
