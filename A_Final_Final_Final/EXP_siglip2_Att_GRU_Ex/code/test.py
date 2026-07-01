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
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

NUM_PHASE1_SAMPLES = 10
import random

warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["PYTORCH_MULTIPROCESSING_REDIRECTS"] = "0"
logging.getLogger("torch").setLevel(logging.ERROR)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_PATH = os.path.join(MODELS_DIR, "BanglaGPT")
os.environ["HF_HOME"] = os.path.join(MODEL_FOLDER, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total_vram - 256e6) / total_vram)
    torch.cuda.empty_cache()

MAX_SEQ_LEN = 26
BATCH_SIZE = 8  # Vectorized batch processing will now scale VRAM usage with this number
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.1
Num_Workers = 1
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05
TRAIN_SEED = 8316

GRU_NUM_LAYERS = 4
LSTM_NUM_LAYERS = 4
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
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, encoder_outputs, input_ids, hidden=None):
        batch_size = input_ids.size(0)
        device = input_ids.device
        embeddings = self.dropout(self.embedding(input_ids))
        if hidden is None:
            hidden = torch.zeros(GRU_NUM_LAYERS, batch_size, HIDDEN_DIM, device=device)
        rnn_outputs, hidden = self.rnn(embeddings, hidden)
        context = self.attention(rnn_outputs, encoder_outputs)
        combined = torch.cat([context, rnn_outputs], dim=-1)
        fused = self.tanh(self.concat(combined))
        return self.dropout(fused)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=LSTM_NUM_LAYERS, batch_first=True
        )
        self.attention = AttentionLayer(hidden_dim)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, encoder_outputs, gru_features, hidden=None):
        batch_size = gru_features.size(0)
        device = gru_features.device
        if hidden is None:
            h0 = torch.zeros(LSTM_NUM_LAYERS, batch_size, HIDDEN_DIM, device=device)
            c0 = torch.zeros(LSTM_NUM_LAYERS, batch_size, HIDDEN_DIM, device=device)
            hidden = (h0, c0)
        rnn_outputs, hidden = self.rnn(gru_features, hidden)
        context = self.attention(rnn_outputs, encoder_outputs)
        combined = torch.cat([context, rnn_outputs], dim=-1)
        fused = self.tanh(self.concat(combined))
        return self.dropout(fused)


class BanglaGPTDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.input_proj = nn.Linear(HIDDEN_DIM + EMBED_DIM, BANGLA_GPT_HIDDEN)
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

    def forward(self, lstm_features, input_ids):
        text_embeds = self.embedding(input_ids)
        combined = torch.cat([lstm_features, text_embeds], dim=-1)
        gpt_inputs = self.input_proj(combined)
        gpt_out = self.gpt(inputs_embeds=gpt_inputs).last_hidden_state
        return self.output_proj(self.dropout(gpt_out))


class MultiDecoderFusion(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.gru = GRUDecoder(vocab_size)
        self.lstm = LSTMDecoder()
        self.banglagpt = BanglaGPTDecoder(vocab_size)

    def forward(self, encoder_outputs, input_ids):
        gru_features = self.gru(encoder_outputs, input_ids)
        lstm_features = self.lstm(encoder_outputs, gru_features)
        return self.banglagpt(lstm_features, input_ids)


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
        self.decoder = MultiDecoderFusion(vocab_size)

    def forward(self, pixel_values, input_ids):
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        return self.decoder(visual_tokens, input_ids)

    @torch.no_grad()
    def generate_beam_batch(
        self, pixel_values, vocab, beam_size=1, max_new_tokens=MAX_SEQ_LEN
    ):
        """Highly optimized batched beam search processing sequence generations in parallel."""
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)

        start_id = vocab.word2idx["<start>"]
        end_id = vocab.word2idx["<end>"]

        # Expand vision tokens across the beam dimension
        # Shape: (batch_size * beam_size, num_tokens, hidden_dim)
        visual_tokens = visual_tokens.repeat_interleave(beam_size, dim=0)

        # Tracking variables
        sequences = torch.full(
            (batch_size * beam_size, 1), start_id, dtype=torch.long, device=device
        )
        beam_scores = torch.full((batch_size, beam_size), -1e9, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        done_mask = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens - 1):
            if done_mask.all():
                break

            logits = self.decoder(visual_tokens, sequences)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            # Mask paths that are already finished
            log_probs[done_mask] = -1e9
            log_probs[done_mask, vocab.word2idx["<pad>"]] = 0.0

            # Suppress <unk> token (index 3) so it is never generated
            log_probs[:, vocab.word2idx.get("<unk>", 3)] = -1e9

            # Compute cumulative score
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores = next_scores.view(batch_size, -1)

            topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)

            beam_scores = topk_scores.view(-1)

            beam_ids = topk_indices // vocab.vocab_size
            token_ids = topk_indices % vocab.vocab_size

            batch_offsets = torch.arange(
                0, batch_size * beam_size, beam_size, device=device
            ).unsqueeze(-1)
            flat_beam_ids = (beam_ids + batch_offsets).view(-1)

            sequences = torch.cat(
                [sequences[flat_beam_ids], token_ids.view(-1, 1)], dim=-1
            )
            done_mask = done_mask[flat_beam_ids]

            new_ends = (token_ids == end_id).view(-1)
            done_mask = done_mask | new_ends

        sequences = sequences.view(batch_size, beam_size, -1)
        best_sequences = sequences[:, 0, :]

        return [vocab.decode(seq.tolist()) for seq in best_sequences]


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
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            return None

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            all_caps = self.captions.get(img_name, [""])
            return pixel_values, all_caps, img_name
        except Exception:
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    pixel_values = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    img_names = [item[2] for item in batch]
    return pixel_values, captions, img_names


def calculate_metrics(reference_list, hypothesis):
    hypothesis = hypothesis.replace("।", "")
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    list_of_refs_tokens = [
        nltk.word_tokenize(ref.replace("।", "")) for ref in reference_list
    ]
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
        calculate_rouge_l(nltk.word_tokenize(ref.replace("।", "")), hypothesis_tokens)
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

    hypothesis = hypothesis.replace("।", "")
    hyp_tokens = nltk.word_tokenize(hypothesis)
    scores = []
    for ref_tokens in [
        nltk.word_tokenize(ref.replace("।", "")) for ref in reference_list
    ]:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens):
                continue
            score += cosine_similarity(
                get_ngrams(ref_tokens, n), get_ngrams(hyp_tokens, n)
            )
        scores.append(score / 4.0)
    return max(scores) if scores else 0.0


def run_evaluation():
    print(f"Target Device: {DEVICE}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.count} tokens")

    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))
    image_names = list(captions.keys())
    n = len(image_names)

    generator = torch.Generator().manual_seed(TRAIN_SEED)
    perm = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    test_list = [image_names[i] for i in perm[n_train + n_val :]]

    if not test_list:
        print("Error: no test images in split!")
        return

    print(f"Test split: {len(test_list)} images ({(len(test_list) / n) * 100:.2f}%)")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model_old.pt")
    if not os.path.exists(best_model_path):
        print(f"Best model not found at {best_model_path}")
        return
    print(f"Loading best model: {best_model_path}")

    current_module = sys.modules[__name__]
    sys.modules["__main__"] = current_module
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)

    print("Building model (SigLIP2 → MultiDecoder Fusion)...")
    model = CaptionModel(vocab_size=vocab.vocab_size).to(DEVICE)
    print("Loading trained weights...")

    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        else:
            clean_state_dict[k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
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

    test_dataset = BanglaCaptionDataset(
        test_list, captions, processor, os.path.join(DATA_DIR, "images")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Num_Workers,
        collate_fn=collate_fn,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )

    # =========================================================================
    # PHASE 1: IMMEDIATE TARGET IMAGE INFERENCE
    # =========================================================================
    print(f"\n[PHASE 1] Evaluating {NUM_PHASE1_SAMPLES} random test images...")
    phase1_images = set(
        random.sample(test_list, min(NUM_PHASE1_SAMPLES, len(test_list)))
    )
    phase1_results = []
    phase1_scores_map = {}

    with torch.inference_mode():
        for batch in test_loader:
            if not batch or len(phase1_results) >= len(phase1_images):
                continue
            pixel_values, batch_caps, img_names = batch

            # Find elements belonging to Phase 1 inside the batch
            valid_batch_indices = [
                idx for idx, name in enumerate(img_names) if name in phase1_images
            ]
            if not valid_batch_indices:
                continue

            sub_pixel_values = pixel_values[valid_batch_indices].to(DEVICE)
            # Batched processing execution
            gen_caps = model.generate_beam_batch(sub_pixel_values, vocab, beam_size=1)

            for index, idx in enumerate(valid_batch_indices):
                name = img_names[idx]
                ref_list = batch_caps[idx]
                gen_cap = gen_caps[index].strip()

                m = calculate_metrics(ref_list, gen_cap)
                m["cider"] = calculate_cider(ref_list, gen_cap)

                res_item = {
                    "img_name": name,
                    "gen_caption": gen_cap,
                    "references": ref_list,
                    "scores": m,
                }
                phase1_results.append(res_item)
                phase1_scores_map[name] = m

    if phase1_results:
        print("\n" + "=" * 100)
        print(
            f"{'Image':<12} {'BLEU-1':<8} {'BLEU-2':<8} {'BLEU-3':<8} {'BLEU-4':<8} {'METEOR':<8} {'ROUGE-L':<8} {'CIDEr':<8}  Caption"
        )
        print("-" * 100)
        for r in phase1_results:
            s = r["scores"]
            cap_short = r["gen_caption"][:40]
            print(
                f"{r['img_name']:<12} {s['bleu1']:<8.4f} {s['bleu2']:<8.4f} {s['bleu3']:<8.4f} {s['bleu4']:<8.4f} {s['meteor']:<8.4f} {s['rouge_l']:<8.4f} {s['cider']:<8.4f}  {cap_short}"
            )
        print("=" * 100)

    # =========================================================================
    # PHASE 2: GLOBAL FULL AVERAGES (Completely Batch-Vectorized)
    # =========================================================================
    print(
        f"\n[PHASE 2] Starting full evaluation partition over {len(test_list)} images using 6 workers..."
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

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Processing Entire Test Dataset"):
            if not batch:
                continue
            pixel_values, batch_caps, img_names = batch
            pixel_values = pixel_values.to(DEVICE)

            eval_indices = []
            for idx, name in enumerate(img_names):
                if name in phase1_scores_map:
                    for k in all_metrics:
                        all_metrics[k].append(phase1_scores_map[name][k])
                else:
                    eval_indices.append(idx)

            if not eval_indices:
                continue

            # Batch process remaining data tokens simultaneously
            sub_pixel_values = pixel_values[eval_indices]
            gen_caps = model.generate_beam_batch(sub_pixel_values, vocab, beam_size=1)

            for run_idx, idx in enumerate(eval_indices):
                ref_list = batch_caps[idx]
                gen_cap = gen_caps[run_idx].strip()

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
