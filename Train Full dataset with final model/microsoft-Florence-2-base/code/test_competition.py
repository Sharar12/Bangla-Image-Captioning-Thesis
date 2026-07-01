import os
import sys
import io
import random
import torch
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoProcessor,
    Florence2ForConditionalGeneration,
    CLIPImageProcessor,
)
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL_FOLDER))

DATA_DIR = (
    r"D:\Python Projects\Train Full dataset with final model\Final_Hybrid_Dataset_60k"
)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_NAME = "microsoft/Florence-2-base"
HF_TOKEN = "YOUR_HF_TOKEN"
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 32
BATCH_SIZE = 16

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

    def encode(self, sentence, max_len):
        tokens = [
            self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sentence.split()
        ]
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
        if len(tokens) < max_len:
            tokens += [self.word2idx["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[: max_len - 1] + [self.word2idx["<EOS>"]]
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
        full_model = Florence2ForConditionalGeneration.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        self.vision_tower = full_model.model.vision_tower

        for param in self.vision_tower.parameters():
            param.requires_grad = False

        self.feature_dim = 1024

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.vision_tower(pixel_values)
            features = outputs.last_hidden_state

            if len(features.shape) == 4:
                b, c, h, w = features.shape
                features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

            return features


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        hidden_dim=2048,
        vocab_size=10000,
        feature_dim=1024,
        num_layers=6,
        nhead=16,
        dropout=0.1,
    ):
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
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
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

        embed = self.embedding(captions) * (self.embed_dim**0.5)
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

        generated = torch.full(
            (batch_size, 1), self.word2idx["<SOS>"], dtype=torch.long
        ).to(features.device)

        for _ in range(max_len - 1):
            embed = self.embedding(generated) * (self.embed_dim**0.5)
            embed = embed + self.pos_encoder[:, : generated.size(1), :]

            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(
                features.device
            )

            decoder_output = self.decoder(embed, memory, tgt_mask=tgt_mask)
            out = self.fc_out(decoder_output)

            next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.word2idx["<EOS>"]).all():
                break

        return generated

    def beam_search_decode(self, features, beam_width=3, max_len=MAX_SEQ_LEN):
        batch_size = features.size(0)
        device = features.device

        memory = self.fc_feature(features)
        memory = self.encoder(memory)

        final_captions = []

        for b in range(batch_size):
            start_token = self.word2idx["<SOS>"]
            beams = [([start_token], 0.0)]

            for _ in range(max_len - 1):
                candidates = []
                for seq, score in beams:
                    if seq[-1] == self.word2idx["<EOS>"]:
                        candidates.append((seq, score))
                        continue

                    tgt = torch.tensor([seq]).to(device)
                    embed = self.embedding(tgt) * (self.embed_dim**0.5)
                    embed = embed + self.pos_encoder[:, : tgt.size(1), :]

                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(
                        device
                    )
                    mem_slice = memory[b : b + 1]

                    decoder_output = self.decoder(embed, mem_slice, tgt_mask=tgt_mask)
                    logits = self.fc_out(decoder_output[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)

                    top_probs, top_idx = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        candidates.append(
                            (
                                seq + [top_idx[0, i].item()],
                                score + top_probs[0, i].item(),
                            )
                        )

                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[
                    :beam_width
                ]
                if all(s[-1] == self.word2idx["<EOS>"] for s, sc in beams):
                    break

            final_captions.append(beams[0][0])

        return torch.tensor(
            [seq + [0] * (max_len - len(seq)) for seq in final_captions],
            dtype=torch.long,
        )


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

    def beam_search_decode(self, features, beam_width=3, max_len=MAX_SEQ_LEN):
        return self.decoder.beam_search_decode(features, beam_width, max_len)


def calculate_metrics(reference_list, hypothesis):
    hypothesis_tokens = hypothesis.split()
    list_of_refs_tokens = [ref.split() for ref in reference_list]

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
    except:
        meteor = 0.0

    rouge_scores = [
        calculate_rouge_l(ref.split(), hypothesis_tokens) for ref in reference_list
    ]
    rouge_l = max(rouge_scores)

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": rouge_l,
    }


def calculate_rouge_l(reference, hypothesis):
    ref_set = set(reference)
    hyp_set = set(hypothesis)

    if len(ref_set) == 0 or len(hyp_set) == 0:
        return 0.0

    overlap = len(ref_set.intersection(hyp_set))

    precision = overlap / len(hyp_set) if len(hyp_set) > 0 else 0
    recall = overlap / len(ref_set) if len(ref_set) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_cider(reference_list, hypothesis):
    def get_ngrams(tokens, n):
        return list(ngrams(tokens, n))

    def cosine_similarity(ref_ngrams, hyp_ngrams, ref_length, hyp_length):
        ref_freq = defaultdict(int)
        hyp_freq = defaultdict(int)

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

        dot_product = sum(ref_vec[i] * hyp_vec[i] for i in range(len(all_ngrams)))
        return dot_product / (ref_norm * hyp_norm)

    hyp_tokens = hypothesis.split()
    ref_tokens_list = [ref.split() for ref in reference_list]

    scores = []
    for ref_tokens in ref_tokens_list:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens):
                continue

            hyp_ngrams = get_ngrams(hyp_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)

            sim = cosine_similarity(
                ref_ngrams, hyp_ngrams, len(ref_tokens), len(hyp_tokens)
            )
            score += sim

        scores.append(score / 4.0)

    return max(scores) if scores else 0.0


def test():
    print(f"Device: {DEVICE}")
    print("Loading data...")

    test_list = load_image_list(os.path.join(DATA_DIR, "test.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))

    print(f"Test images: {len(test_list)}")
    print(f"Captioned images: {len(captions)}")

    checkpoint = torch.load(
        os.path.join(OUTPUT_DIR, "checkpoint_epoch_10.pt"), map_location=DEVICE
    )
    word2idx = checkpoint["vocab"]
    vocab = Vocabulary(word2idx)
    print(f"Vocabulary size: {vocab.n_words}")

    model = CaptionModel(vocab.n_words, word2idx).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.decoder.word2idx = word2idx

    processor = CLIPImageProcessor.from_pretrained(MODEL_DIR)

    test_dataset = BanglaCaptionDataset(
        test_list, captions, vocab, processor, os.path.join(DATA_DIR, "images")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("\nEvaluating...")
    all_results = []

    with torch.inference_mode():
        for pixel_values, tokens, img_names in tqdm(test_loader, desc="Testing"):
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda"):
                features = model.encoder(pixel_values)
                predictions = model.decoder.greedy_decode(features, max_len=MAX_SEQ_LEN)

            preds_cpu = predictions.cpu().numpy()
            for i, img_name in enumerate(img_names):
                generated_caption = vocab.decode(preds_cpu[i])
                ref_list = captions.get(img_name, [])

                if ref_list:
                    all_results.append((img_name, ref_list, generated_caption))

    print("\nCalculating Metrics...")
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
        cider_score = calculate_cider(ref_list, hyp)
        m["cider"] = cider_score

        for k in all_metrics:
            all_metrics[k].append(m[k])

        detailed_results.append(
            {"img_name": img_name, "reference": ref_list, "hypothesis": hyp, **m}
        )

    avg_cider = (
        sum(all_metrics["cider"]) / len(all_metrics["cider"])
        if all_metrics["cider"]
        else 0.0
    )
    print(f"CIDER     : {avg_cider:.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("FINAL RESEARCH SCORES")
    print("=" * 60)

    for metric, scores in all_metrics.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"{metric.upper():10s}: {avg:.4f}")

    print("=" * 60)

    results_summary = {
        metric: sum(scores) / len(scores) if scores else 0
        for metric, scores in all_metrics.items()
    }
    results_summary["samples"] = len(all_results)

    pd.DataFrame([results_summary]).to_csv(
        os.path.join(OUTPUT_DIR, "test_results.csv"), index=False
    )
    pd.DataFrame(detailed_results).to_csv(
        os.path.join(OUTPUT_DIR, "detailed_results.csv"), index=False
    )

    random_idx = random.randint(0, len(detailed_results) - 1)
    sample = detailed_results[random_idx]

    img_path = os.path.join(DATA_DIR, "images", sample["img_name"])
    img = Image.open(img_path).convert("RGB")
    img.save(os.path.join(OUTPUT_DIR, "random_sample.jpg"))

    print(f"\n{'=' * 70}")
    print(f"RANDOM SAMPLE (Index: {random_idx})")
    print(f"{'=' * 70}")
    print(f"Image: {sample['img_name']}")
    print(f"Generated: {sample['hypothesis']}")
    print(f"Actual:    {sample['reference']}")
    print(f"{'-' * 70}")
    print(f"Scores:")
    print(f"  BLEU-1:  {sample['bleu1']:.4f}")
    print(f"  BLEU-2:  {sample.get('bleu2', 0):.4f}")
    print(f"  BLEU-3:  {sample.get('bleu3', 0):.4f}")
    print(f"  BLEU-4:  {sample.get('bleu4', 0):.4f}")
    print(f"  METEOR:  {sample['meteor']:.4f}")
    print(f"  ROUGE-L: {sample['rouge_l']:.4f}")
    print(f"  CIDER:   {sample.get('cider', avg_cider):.4f}")
    print(f"{'=' * 70}")
    print(f"Image saved to: {os.path.join(OUTPUT_DIR, 'random_sample.jpg')}")


if __name__ == "__main__":
    test()
