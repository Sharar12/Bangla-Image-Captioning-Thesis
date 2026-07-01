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

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BANGLA_SUFFIXES = [
    "\u0997\u09c1\u09b2\u09cb",
    "\u0997\u09c1\u09b2\u09bf",
    "\u0997\u09a3",
    "\u09ac\u09cd\u09af\u09be\u0995\u09cd\u099f\u09c7\u09a8\u09bf",
    "\u099b\u09bf\u09b2\u09be\u09ae",
    "\u099b\u09bf\u09b2\u09c7",
    "\u099b\u09bf\u09b2\u09be",
    "\u099b\u09bf\u09b2",
    "\u099b\u09bf\u09b2\u09c7\u09a8",
    "\u099b\u09bf\u09b2\u09be\u09ae",
    "\u09be\u099a\u09cd\u099b\u09c7",
    "\u09be\u099a\u09cd\u099b\u09bf",
    "\u09be\u099a\u09cd\u099b\u09c7\u09a8",
    "\u099b\u09c7",
    "\u099b\u09bf",
    "\u099b\u09c7\u09a8",
    "\u09b2\u09be\u09ae",
    "\u09b2\u09c7\u09a8",
    "\u09b2\u09c7",
    "\u09b2\u09be",
    "\u09b2",
    "\u09ac\u09c7\u09a8",
    "\u09ac\u09c7",
    "\u09ac",
    "\u09a6\u09cd\u09ac\u09be\u09b0\u09be",
    "\u09a6\u09bf\u09df\u09c7",
    "\u09a5\u09c7\u0995\u09c7",
    "\u0995\u09c7",
    "\u09b0",
    "\u09a4\u09c7",
    "\u09df",
    "\u098f",
    "\u0996\u09be\u09a8\u09be",
    "\u0996\u09be\u09a8\u09bf",
    "\u099f\u09be",
    "\u099f\u09bf",
    "\u099f\u09be\u0987",
    "\u099f\u09be\u09df",
    "\u099f\u09bf\u09b0",
    "\u099f\u09be\u09b0",
    "\u099f\u09be\u09a4\u09c7",
    "\u09a6\u09c7\u09b0",
]

_BPUNCT = '\\[\\](){}""' + "\u2018\u2019\u201c\u201d" + "\u0964\u0965" + "!?,;:.-"
BANGLA_PUNCT_PATTERN = re.compile(f"^[{_BPUNCT}]+|[{_BPUNCT}]+$")


def normalize_bangla(text):
    text = unicodedata.normalize("NFC", text)
    return text.replace("\u200c", "").replace("\u200d", "").strip()


def tokenize_bangla(text):
    tokens = []
    for word in normalize_bangla(text).split():
        word = BANGLA_PUNCT_PATTERN.sub("", word)
        if word:
            tokens.append(word)
    return tokens


def stem_bangla_word(word):
    if len(word) < 3:
        return word
    for suffix in sorted(BANGLA_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


def stem_bangla(tokens):
    return [stem_bangla_word(t) for t in tokens]


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


class BanglaLuongAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attn_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT)

    def _attention(self, decoder_output, encoder_outputs):
        adapted = self.attn_W(encoder_outputs)
        scores = torch.bmm(decoder_output, adapted.transpose(1, 2))
        alignment = torch.softmax(scores, dim=-1)
        context = torch.bmm(alignment, encoder_outputs)
        return context

    def forward(self, encoder_outputs, input_ids, hidden=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        embeddings = self.dropout(self.embedding(input_ids))
        if hidden is None:
            hidden = torch.zeros(1, batch_size, HIDDEN_DIM, device=device)
        decoder_outputs, hidden = self.gru(embeddings, hidden)
        adapted = self.attn_W(encoder_outputs)
        scores = torch.bmm(decoder_outputs, adapted.transpose(1, 2))
        alignment = torch.softmax(scores, dim=-1)
        context = torch.bmm(alignment, encoder_outputs)
        combined = torch.cat([context, decoder_outputs], dim=-1)
        fused = self.tanh(self.concat(combined))
        logits = self.proj(self.dropout(fused))
        return logits

    def step(self, token_embed, hidden, encoder_outputs):
        decoder_output, hidden = self.gru(token_embed, hidden)
        context = self._attention(decoder_output, encoder_outputs)
        combined = torch.cat([context, decoder_output], dim=-1)
        fused = self.tanh(self.concat(combined))
        logits = self.proj(fused)
        return logits, hidden


class Encoder(nn.Module):
    def forward(self, pixel_values):
        vision_attr = [v for k, v in self._modules.items() if "siglip" in k]
        if vision_attr:
            return vision_attr[0].vision_model(pixel_values).last_hidden_state
        return (
            getattr(self, list(self._modules.keys())[0])
            ._modules["siglip"](pixel_values)
            .last_hidden_state
        )


class CaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = BanglaLuongAttentionDecoder(vocab_size=vocab_size)

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
        encoder_outputs = self.vision_projection(features)
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        start_id, end_id = vocab.word2idx["<start>"], vocab.word2idx["<end>"]
        hidden = None
        generated = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=device
        )
        for _ in range(max_new_tokens):
            token_embed = self.decoder.embedding(generated[:, -1:])
            logits, hidden = self.decoder.step(token_embed, hidden, encoder_outputs)
            next_token = logits.argmax(dim=-1)
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

    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt")
    checkpoints = [
        (int(m.group(1)), os.path.join(OUTPUT_DIR, m.group(0)))
        for f in os.listdir(OUTPUT_DIR)
        if (m := pattern.match(f))
    ]
    if not checkpoints:
        print(f"No checkpoints found in {OUTPUT_DIR}")
        return
    checkpoints.sort(key=lambda x: x[0])
    checkpoint_epoch, checkpoint_path = checkpoints[-1]
    print(f"Loading checkpoint: {checkpoint_path} (epoch {checkpoint_epoch})")

    current_module = sys.modules[__name__]
    sys.modules["__main__"] = current_module
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    print("Building model (SigLIP2 -> Luong Attention + GRU)...")
    model = CaptionModel(vocab_size=vocab.vocab_size)

    siglip_path = (
        SIGLIP_MODEL_PATH
        if os.path.exists(SIGLIP_MODEL_PATH)
        else "google/siglip2-base-patch32-256"
    )
    print("Loading SigLIP2 vision encoder...")
    siglip = AutoModel.from_pretrained(siglip_path, trust_remote_code=True)
    for param in siglip.parameters():
        param.requires_grad = False
    setattr(model.encoder, "siglip", siglip)

    model = model.to(DEVICE)
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
        siglip_path
        if os.path.exists(os.path.join(siglip_path, "preprocessor_config.json"))
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
