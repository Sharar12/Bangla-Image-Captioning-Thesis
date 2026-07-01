import os
import json
import re
import sys
import gc
import random
import warnings

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["PYTORCH_MULTIPROCESSING_REDIRECTS"] = "0"
import logging

logging.getLogger("torch").setLevel(logging.ERROR)

import io
import math
import torch

_RUNNING_UNDER_PYTEST = "pytest" in sys.modules or any(
    "pytest" in os.path.basename(a).lower() for a in sys.argv if os.path.basename(a)
)
if not _RUNNING_UNDER_PYTEST:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image, ImageTk
from collections import defaultdict
import tkinter as tk
from tqdm import tqdm
import unicodedata
import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")

DATA_DIR = r"D:\Python Projects\Train Full dataset with final model - For Thesis\Final_Hybrid_Dataset_60k"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
PARENT_DIR = os.path.dirname(MODEL_FOLDER)
BANGLABERT_DIR = os.path.join(PARENT_DIR, "siglip2_banglabert")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(BANGLABERT_DIR, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_BERT_PATH = os.path.join(MODELS_DIR, "banglabert")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total_vram - 1e9) / total_vram)
    torch.cuda.empty_cache()

MAX_SEQ_LEN = 31
BATCH_SIZE = 64
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.3

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
        return pixel_values, random.choice(all_caps), img_name


def calculate_metrics(reference_list, hypothesis):
    hypothesis_tokens = tokenize_bangla(hypothesis)
    list_of_refs_tokens = [tokenize_bangla(ref) for ref in reference_list]
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
    hyp_stemmed = stem_bangla(hypothesis_tokens)
    refs_stemmed = [stem_bangla(tokenize_bangla(ref)) for ref in reference_list]
    try:
        meteor = meteor_score(refs_stemmed, hyp_stemmed)
    except Exception:
        meteor = 0.0
    rouge_scores = [
        calculate_rouge_l(tokenize_bangla(ref), hypothesis_tokens)
        for ref in reference_list
    ]
    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": max(rouge_scores),
    }


def calculate_rouge_l(reference, hypothesis):
    ref_set, hyp_set = set(reference), set(hypothesis)
    if not ref_set or not hyp_set:
        return 0.0
    overlap = len(ref_set & hyp_set)
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return (
        2 * precision * recall / (precision + recall)
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

    hyp_tokens = tokenize_bangla(hypothesis)
    scores = []
    for ref_tokens in [tokenize_bangla(ref) for ref in reference_list]:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens):
                continue
            score += cosine_similarity(
                get_ngrams(ref_tokens, n), get_ngrams(hyp_tokens, n)
            )
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
    tk.Label(
        text_frame,
        text=f"{img_name}",
        font=("Arial", 12, "bold"),
        fg="#e2e8f0",
        bg="#1e293b",
    ).pack(anchor="w")
    tk.Label(
        text_frame,
        text="\nGenerated Caption:",
        font=("Arial", 11, "bold"),
        fg="#34d399",
        bg="#1e293b",
    ).pack(anchor="w")
    tk.Label(
        text_frame,
        text=generated_caption,
        wraplength=400,
        justify="left",
        fg="#f1f5f9",
        bg="#1e293b",
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)
    tk.Label(
        text_frame,
        text="\nGround Truth References:",
        font=("Arial", 11, "bold"),
        fg="#60a5fa",
        bg="#1e293b",
    ).pack(anchor="w")
    for i, ref in enumerate(ref_list, 1):
        tk.Label(
            text_frame,
            text=f"{i}. {ref}",
            wraplength=400,
            justify="left",
            fg="#94a3b8",
            bg="#1e293b",
        ).pack(anchor="w", pady=1)
    scores_text = (
        f"\nMetrics Matrix:\n  BLEU-1: {scores['bleu1']:.4f}   BLEU-2: {scores['bleu2']:.4f}\n"
        f"  BLEU-3: {scores['bleu3']:.4f}   BLEU-4: {scores['bleu4']:.4f}\n"
        f"  METEOR: {scores['meteor']:.4f}   ROUGE-L: {scores['rouge_l']:.4f}\n"
        f"  CIDEr: {scores.get('cider', 0):.4f}"
    )
    tk.Label(
        text_frame,
        text=scores_text,
        justify="left",
        fg="#c084fc",
        bg="#1e293b",
        font=("Consolas", 10),
    ).pack(anchor="w", pady=5)
    tk.Button(
        root,
        text="Exit Window",
        command=root.destroy,
        font=("Arial", 10),
        bg="#334155",
        fg="white",
        padx=20,
        pady=5,
    ).pack(pady=(0, 15))
    root.mainloop()


def test():
    print(f"Target Device: {DEVICE}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.count} tokens")

    test_txt = os.path.join(DATA_DIR, "test.txt")
    if not os.path.exists(test_txt):
        print("No test.txt found!")
        return
    test_list = load_image_list(test_txt)
    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))

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

    print("Building model (SigLIP2 → Luong Attention + GRU)...")
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

    test_dataset = BanglaCaptionDataset(
        test_list, captions, processor, os.path.join(DATA_DIR, "images")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    gc.collect()
    torch.cuda.empty_cache()

    print("\nRunning inference...")
    all_results = []
    with torch.inference_mode():
        for pixel_values, _, img_names in tqdm(test_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)
            gen_captions = model.generate_captions(
                pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN
            )
            for img_name, gen_cap in zip(img_names, gen_captions):
                ref_list = captions.get(img_name, [])
                if ref_list:
                    all_results.append((img_name, ref_list, gen_cap))

    print("\nCalculating metrics...")
    all_metrics = {
        "bleu1": [],
        "bleu2": [],
        "bleu3": [],
        "bleu4": [],
        "meteor": [],
        "rouge_l": [],
        "cider": [],
    }
    detailed_results = []
    for img_name, ref_list, hyp in tqdm(all_results, desc="Metrics"):
        m = calculate_metrics(ref_list, hyp)
        m["cider"] = calculate_cider(ref_list, hyp)
        for k in all_metrics:
            all_metrics[k].append(m[k])
        detailed_results.append(
            {"img_name": img_name, "reference": ref_list, "hypothesis": hyp, **m}
        )

    print(
        "\n"
        + "=" * 50
        + "\nFINAL EVALUATION SCORES (Att-GRU — Luong concat)\n"
        + "=" * 50
    )
    for metric, scores in all_metrics.items():
        print(f"{metric.upper():10s}: {sum(scores) / len(scores):.4f}")
    print("=" * 50)

    pd.DataFrame([{k: sum(v) / len(v) for k, v in all_metrics.items()}]).to_csv(
        os.path.join(OUTPUT_DIR, "test_results.csv"), index=False
    )
    pd.DataFrame(detailed_results).to_csv(
        os.path.join(OUTPUT_DIR, "detailed_results.csv"), index=False
    )

    random_sample = random.choice(detailed_results)
    img_path = os.path.join(DATA_DIR, "images", random_sample["img_name"])
    if not _RUNNING_UNDER_PYTEST:
        show_sample_popup(
            img_path=img_path,
            generated_caption=random_sample["hypothesis"],
            ref_list=random_sample["reference"],
            scores={
                k: random_sample[k]
                for k in [
                    "bleu1",
                    "bleu2",
                    "bleu3",
                    "bleu4",
                    "meteor",
                    "rouge_l",
                    "cider",
                ]
            },
            img_name=random_sample["img_name"],
        )


def test_main():
    test()


if __name__ == "__main__":
    test()
