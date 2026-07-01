import os
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
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModel,
    BertLMHeadModel,
    BertConfig,
)
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
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_BERT_PATH = os.path.join(MODELS_DIR, "banglabert")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total_vram - 1e9) / total_vram)
    torch.cuda.empty_cache()
MAX_SEQ_LEN = 64
BATCH_SIZE = 64

FEATURE_DIM = 768
HIDDEN_SIZE = 768
DROPOUT = 0.1

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Bangla text preprocessing utilities
BANGLA_SUFFIXES = [
    # Plural markers
    "\u0997\u09c1\u09b2\u09cb",
    "\u0997\u09c1\u09b2\u09bf",
    "\u0997\u09a3",
    # Verb endings (past perfect)
    "\u09ac\u09cd\u09af\u09be\u0995\u09cd\u099f\u09c7\u09a8\u09bf",
    "\u099b\u09bf\u09b2\u09be\u09ae",
    "\u099b\u09bf\u09b2\u09c7",
    "\u099b\u09bf\u09b2\u09be",
    "\u099b\u09bf\u09b2",
    "\u099b\u09bf\u09b2\u09c7\u09a8",
    "\u099b\u09bf\u09b2\u09be\u09ae",
    # Verb endings (present continuous)
    "\u09be\u099a\u09cd\u099b\u09c7",
    "\u09be\u099a\u09cd\u099b\u09bf",
    "\u09be\u099a\u09cd\u099b\u09c7\u09a8",
    "\u099b\u09c7",
    "\u099b\u09bf",
    "\u099b\u09c7\u09a8",
    # Verb endings (past simple)
    "\u09b2\u09be\u09ae",
    "\u09b2\u09c7\u09a8",
    "\u09b2\u09c7",
    "\u09b2\u09be",
    "\u09b2",
    # Verb endings (future/habitual)
    "\u09ac\u09c7\u09a8",
    "\u09ac\u09c7",
    "\u09ac",
    # Postpositions
    "\u09a6\u09cd\u09ac\u09be\u09b0\u09be",
    "\u09a6\u09bf\u09df\u09c7",
    "\u09a5\u09c7\u0995\u09c7",
    # Case endings
    "\u0995\u09c7",
    "\u09b0",
    "\u09a4\u09c7",
    "\u09df",
    "\u098f",
    # Definite articles
    "\u0996\u09be\u09a8\u09be",
    "\u0996\u09be\u09a8\u09bf",
    "\u099f\u09be",
    "\u099f\u09bf",
    # Emphatic/compound forms
    "\u099f\u09be\u0987",
    "\u099f\u09be\u09df",
    "\u099f\u09bf\u09b0",
    "\u099f\u09be\u09b0",
    "\u099f\u09be\u09a4\u09c7",
    "\u09a6\u09c7\u09b0",
]

# Strip leading/trailing punctuation from Bangla words
_BPUNCT = (
    '\\[\\](){}""'  # brackets and quotes
    "\u2018\u2019\u201c\u201d"  # curly quotes
    "\u0964\u0965"  # Bangla danda, double danda
    "!?,;:.-"
)
BANGLA_PUNCT_PATTERN = re.compile(f"^[{_BPUNCT}]+|[{_BPUNCT}]+$")
BANGLA_PUNCT_PATTERN = re.compile(f"^[{_BPUNCT}]+|[{_BPUNCT}]+$")


def normalize_bangla(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    return text.strip()


def tokenize_bangla(text):
    text = normalize_bangla(text)
    tokens = []
    for word in text.split():
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


class Encoder(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.feature_dim = feature_dim

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
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
        )

    def forward(self, pixel_values, input_ids, attention_mask=None):
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        text_embeds = self.banglabert.get_input_embeddings()(input_ids)

        visual_tokens = visual_tokens.to(dtype=text_embeds.dtype)
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        batch_size, seq_len = input_ids.shape
        num_vis = visual_tokens.size(1)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, num_vis + seq_len, device=input_ids.device
            )
        else:
            vis_mask = torch.ones(
                batch_size, num_vis, device=input_ids.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        outputs = self.banglabert(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    @torch.no_grad()
    def generate_captions(self, pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)

        bos_id = tokenizer.cls_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id

        bos_embed = self.banglabert.get_input_embeddings()(
            torch.full(
                (visual_tokens.size(0), 1),
                bos_id,
                dtype=torch.long,
                device=visual_tokens.device,
            )
        )
        inputs_embeds = torch.cat([visual_tokens, bos_embed], dim=1)

        outputs = self.banglabert.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            return_dict_in_generate=True,
        )

        sequences = outputs.sequences
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]

    @torch.no_grad()
    def generate_caption(self, pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN):
        return self.generate_captions(pixel_values, tokenizer, max_new_tokens)[0]


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
    # Pre-stemmed tokens for METEOR (recommended for Bangla)
    hyp_stemmed = stem_bangla(hypothesis_tokens)
    refs_stemmed = [stem_bangla(tokenize_bangla(ref)) for ref in reference_list]
    try:
        meteor = meteor_score(refs_stemmed, hyp_stemmed)
    except:
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
    if len(ref_set) == 0 or len(hyp_set) == 0:
        return 0.0
    overlap = len(ref_set.intersection(hyp_set))
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return (
        2 * (precision * recall) / (precision + recall)
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
        f"\nMetrics Matrix:\n"
        f"  BLEU-1: {scores['bleu1']:.4f}   BLEU-2: {scores['bleu2']:.4f}\n"
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

    test_txt = os.path.join(DATA_DIR, "test.txt")
    if not os.path.exists(test_txt):
        print("No test.txt found!")
        return
    test_list = load_image_list(test_txt)

    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))

    # Auto-select the checkpoint with the highest epoch number
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

    print("Building model architecture...")
    model = CaptionModel()

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

    print("Loading BanglaBERT decoder...")
    config = BertConfig.from_pretrained(
        BANGLA_BERT_PATH
        if os.path.exists(BANGLA_BERT_PATH)
        else "csebuetnlp/banglabert",
        trust_remote_code=True,
    )
    config.is_decoder = True
    banglabert = BertLMHeadModel.from_pretrained(
        BANGLA_BERT_PATH
        if os.path.exists(BANGLA_BERT_PATH)
        else "csebuetnlp/banglabert",
        config=config,
        trust_remote_code=True,
    )
    for param in banglabert.parameters():
        param.requires_grad = False
    setattr(model, "banglabert", banglabert)

    model = model.to(DEVICE)

    print("Loading trained weights...")
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"Missing keys (not in checkpoint): {len(missing)}")
    if unexpected:
        print(f"Unexpected keys (not in model): {len(unexpected)}")
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

    print("Loading BanglaBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BANGLA_BERT_PATH
        if os.path.exists(BANGLA_BERT_PATH)
        else "csebuetnlp/banglabert",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
                pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN
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

    print("\n" + "=" * 50 + "\nFINAL TRANSLATION SCORES\n" + "=" * 50)
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
