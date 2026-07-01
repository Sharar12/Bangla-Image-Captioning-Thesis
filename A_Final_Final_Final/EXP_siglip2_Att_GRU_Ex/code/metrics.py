import os
import json
import sys
import gc
import math
import warnings
import logging
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

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_PATH = os.path.join(MODELS_DIR, "BanglaGPT")
os.environ["HF_HOME"] = os.path.join(MODEL_FOLDER, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 67
BATCH_SIZE = 64
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.1
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05
TRAIN_SEED = 42
GRU_NUM_LAYERS = 4
LSTM_NUM_LAYERS = 4
BANGLA_GPT_HIDDEN = 768

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        self.gate = nn.Linear(HIDDEN_DIM, 3)

    def forward(self, encoder_outputs, input_ids):
        logits_gru = self.gru(encoder_outputs, input_ids)
        logits_lstm = self.lstm(encoder_outputs, input_ids)
        logits_gpt = self.banglagpt(encoder_outputs, input_ids)
        w = torch.softmax(self.gate(encoder_outputs.mean(1)), dim=-1)
        w = w.unsqueeze(-1).unsqueeze(-1)
        return w[:, 0] * logits_gru + w[:, 1] * logits_lstm + w[:, 2] * logits_gpt


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
            token_gru = self.decoder.gru.embedding.weight[generated[:, -1:]]
            logits_g, gru_hidden = self.decoder.gru.step(
                token_gru, gru_hidden, visual_tokens
            )

            token_lstm = self.decoder.lstm.embedding.weight[generated[:, -1:]]
            logits_l, lstm_hidden = self.decoder.lstm.step(
                token_lstm, lstm_hidden, visual_tokens
            )

            vis_gpt = self.decoder.banglagpt.vis_proj(visual_tokens)
            text_embeds_gpt = self.decoder.banglagpt.gpt_proj(
                self.decoder.banglagpt.embedding(generated)
            )
            comb_gpt = torch.cat([vis_gpt, text_embeds_gpt], dim=1)
            gpt_out = self.decoder.banglagpt.gpt(
                inputs_embeds=comb_gpt
            ).last_hidden_state
            logits_b = self.decoder.banglagpt.output_proj(gpt_out[:, -1:])

            w = torch.softmax(self.decoder.gate(visual_tokens.mean(1)), dim=-1)
            avg_logits = (
                w[:, 0:1, None] * logits_g
                + w[:, 1:2, None] * logits_l
                + w[:, 2:3, None] * logits_b
            )
            next_token = avg_logits.squeeze(1).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return [vocab.decode(seq.tolist()) for seq in generated]


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

    def rouge_l(ref_tokens, hyp_tokens):
        ref_set, hyp_set = set(ref_tokens), set(hyp_tokens)
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

    rouge_scores = [rouge_l(ref, hypothesis_tokens) for ref in list_of_refs_tokens]

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

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": max(rouge_scores) if rouge_scores else 0.0,
        "cider": calculate_cider(reference_list, hypothesis),
    }


def evaluate_split(model, vocab, processor, split_name, image_names, captions):
    if not image_names:
        print(f"  {split_name}: no images, skipping")
        return None

    print(f"\n  Evaluating {split_name} ({len(image_names)} images)...")
    all_metrics = {
        k: []
        for k in ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]
    }
    EVAL_BATCH = 32

    for start_idx in range(0, len(image_names), EVAL_BATCH):
        batch_names = image_names[start_idx : start_idx + EVAL_BATCH]
        batch_images, batch_valid = [], []
        for img_name in batch_names:
            img_path = os.path.join(DATA_DIR, "images", img_name)
            if os.path.exists(img_path):
                batch_images.append(Image.open(img_path).convert("RGB"))
                batch_valid.append(img_name)

        if not batch_images:
            continue

        pixel_values = processor(images=batch_images, return_tensors="pt")[
            "pixel_values"
        ].to(DEVICE)

        with torch.inference_mode():
            gen_caps = model.generate_captions(pixel_values, vocab)
            gen_caps = [c.split("।")[0] + "।" if "।" in c else c for c in gen_caps]

        for img_name, gen_cap in zip(batch_valid, gen_caps):
            ref_list = captions.get(img_name, [])
            if not ref_list:
                continue
            m = calculate_metrics(ref_list, gen_cap)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    averages = {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}
    return averages


def main():
    print(f"Device: {DEVICE}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary: {vocab.count} tokens")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if not os.path.exists(best_model_path):
        print(f"Model not found at {best_model_path}")
        return
    print(f"Loading: {best_model_path}")

    current_module = sys.modules[__name__]
    sys.modules["__main__"] = current_module
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)

    model = CaptionModel(vocab_size=vocab.vocab_size).to(DEVICE)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.eval()

    processor_path = (
        SIGLIP_MODEL_PATH
        if os.path.exists(os.path.join(SIGLIP_MODEL_PATH, "preprocessor_config.json"))
        else "google/siglip2-base-patch32-256"
    )
    processor = AutoImageProcessor.from_pretrained(
        processor_path, trust_remote_code=True
    )

    captions = load_captions(os.path.join(DATA_DIR, "caption.txt"))
    image_names = list(captions.keys())
    n = len(image_names)

    generator = torch.Generator().manual_seed(TRAIN_SEED)
    perm = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_names = [image_names[i] for i in perm[:n_train]]
    val_names = [image_names[i] for i in perm[n_train : n_train + n_val]]
    test_names = [image_names[i] for i in perm[n_train + n_val :]]

    print(f"\n{'=' * 60}")
    print(
        f"{'Split':<12} {'Images':<10} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-3':<10} {'BLEU-4':<10} {'METEOR':<10} {'ROUGE-L':<10} {'CIDEr':<10}"
    )
    print(f"{'=' * 60}")

    all_results = []
    for split_name, split_list in [
        ("Train", train_names),
        ("Val", val_names),
        ("Test", test_names),
    ]:
        gc.collect()
        torch.cuda.empty_cache()
        avg = evaluate_split(model, vocab, processor, split_name, split_list, captions)
        if avg:
            all_results.append({"Split": split_name, **avg})
            print(
                f"{split_name:<12} {len(split_list):<10} {avg['bleu1']:<10.4f} {avg['bleu2']:<10.4f} {avg['bleu3']:<10.4f} {avg['bleu4']:<10.4f} {avg['meteor']:<10.4f} {avg['rouge_l']:<10.4f} {avg['cider']:<10.4f}"
            )

    print(f"{'=' * 60}")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics_split_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'metrics_split_results.csv')}")


if __name__ == "__main__":
    main()
