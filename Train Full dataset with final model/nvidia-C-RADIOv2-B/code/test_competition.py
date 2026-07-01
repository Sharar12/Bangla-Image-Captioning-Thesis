import os
import sys
import io
import random
import torch
import math

# Force UTF-8 for CMD output to handle Bangla characters correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# --- ABSOLUTE PATH INJECTION ---
# This calculates the PROJECT_ROOT (nvidia-C-RADIOv2-B) based on this file's location
CURRENT_FILE_PATH = os.path.abspath(__file__)
CODE_DIR = os.path.dirname(CURRENT_FILE_PATH)
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# Force the Project Root into the system path so 'import model' works
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Attempt the import with detailed error feedback
try:
    from model.hf_model import RADIOModel, RADIOConfig
except ImportError as e:
    print(f"\n[!] CRITICAL IMPORT ERROR: {e}")
    print(f"I am looking for the 'model' folder inside: {PROJECT_ROOT}")
    if os.path.exists(os.path.join(PROJECT_ROOT, "model")):
        print(
            "The 'model' folder exists, but 'hf_model.py' might be missing or named differently."
        )
    else:
        print("The 'model' folder was NOT found in that path.")
    sys.exit(1)

from transformers import CLIPImageProcessor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

# Constants
DATA_DIR = (
    r"D:\Python Projects\Train Full dataset with final model\Final_Hybrid_Dataset_60k"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 32
BATCH_SIZE = 64

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
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"shortest_edge": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
        )
        pixel_values = inputs["pixel_values"].squeeze(0).clone()
        all_caps = self.captions.get(img_name, [""])
        tokens = torch.tensor(
            self.vocab.encode(random.choice(all_caps), MAX_SEQ_LEN), dtype=torch.long
        )
        return pixel_values, tokens, img_name


def custom_collate(batch):
    pvs, tks, names = zip(*batch)
    return torch.stack(pvs, 0), torch.stack(tks, 0), list(names)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = RADIOConfig.from_pretrained(MODEL_DIR)
        self.radio = RADIOModel.from_pretrained(MODEL_DIR, config=config)
        for param in self.radio.parameters():
            param.requires_grad = False
        self.feature_dim = 768
        self.fc_proj = nn.Linear(2304, 768)

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.radio(pixel_values)
        features = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        if features.dim() == 2:
            features = features.unsqueeze(1)
        return self.fc_proj(features)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        hidden_dim=2048,
        vocab_size=10000,
        feature_dim=768,
        num_layers=6,
        nhead=12,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim, self.vocab_size = embed_dim, vocab_size
        self.word2idx = None
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim))
        layer_cfg = {
            "d_model": embed_dim,
            "nhead": nhead,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "batch_first": True,
        }
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_cfg), num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(**layer_cfg), num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_feature = nn.Linear(feature_dim, embed_dim)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def greedy_decode(self, features, max_len=MAX_SEQ_LEN):
        batch_size = features.size(0)
        projected = self.fc_feature(features)
        memory = self.encoder(projected)
        generated = torch.full(
            (batch_size, 1), self.word2idx["<SOS>"], dtype=torch.long
        ).to(features.device)
        for _ in range(max_len - 1):
            embed = (
                self.embedding(generated) * (self.embed_dim**0.5)
                + self.pos_encoder[:, : generated.size(1), :]
            )
            mask = self.generate_square_subsequent_mask(generated.size(1)).to(
                features.device
            )
            out = self.fc_out(self.decoder(embed, memory, tgt_mask=mask))
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


def get_all_metrics(refs, hyp):
    hyp_t = hyp.split()
    ref_t = [r.split() for r in refs]
    smooth = SmoothingFunction().method4
    b1 = sentence_bleu(ref_t, hyp_t, weights=(1, 0, 0, 0), smoothing_function=smooth)
    b2 = sentence_bleu(
        ref_t, hyp_t, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth
    )
    b3 = sentence_bleu(
        ref_t, hyp_t, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth
    )
    b4 = sentence_bleu(
        ref_t, hyp_t, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth
    )
    try:
        met = meteor_score(ref_t, hyp_t)
    except:
        met = 0.0

    def rl_score(r, h):
        rs, hs = set(r), set(h)
        if not rs or not hs:
            return 0.0
        inter = len(rs.intersection(hs))
        p, r_m = inter / len(hs), inter / len(rs)
        return (2 * p * r_m) / (p + r_m) if (p + r_m) > 0 else 0.0

    rl = max([rl_score(r, hyp_t) for r in ref_t]) if ref_t else 0.0

    def cid_sim(r, h):
        s = 0
        for n in [1, 2, 3, 4]:
            hn, rn = set(ngrams(h, n)), set(ngrams(r, n))
            if rn:
                s += len(hn.intersection(rn)) / len(rn)
        return s / 4.0

    cid = max([cid_sim(r.split(), hyp_t) for r in refs]) if refs else 0.0
    return {
        "bleu1": b1,
        "bleu2": b2,
        "bleu3": b3,
        "bleu4": b4,
        "meteor": met,
        "rouge_l": rl,
        "cider": cid,
    }


def test():
    print(f"Device: {DEVICE}")
    test_file = os.path.join(DATA_DIR, "test.txt")
    cap_file = os.path.join(DATA_DIR, "captions.txt")

    with open(test_file, "r") as f:
        test_list = [
            line.strip().split(": ")[-1] if ": " in line else line.strip()
            for line in f
            if line.strip()
        ]

    captions = defaultdict(list)
    with open(cap_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                captions[parts[0]].append(parts[1])

    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_10.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    vocab = Vocabulary(checkpoint["vocab"])

    model = CaptionModel(vocab.n_words, checkpoint["vocab"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    processor = CLIPImageProcessor.from_pretrained(MODEL_DIR)
    dataset = BanglaCaptionDataset(
        test_list, captions, vocab, processor, os.path.join(DATA_DIR, "images")
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=0, collate_fn=custom_collate
    )

    results = []
    detailed_results = []
    with torch.inference_mode():
        for pvs, _, names in tqdm(loader, desc="Testing"):
            pvs = pvs.to(DEVICE)
            with torch.amp.autocast("cuda"):
                features = model.encoder(pvs)
                preds = model.decoder.greedy_decode(features)
            for i, name in enumerate(names):
                hyp = vocab.decode(preds[i].cpu().numpy())
                m = get_all_metrics(captions.get(name, []), hyp)
                results.append(
                    {"img_name": name, "ref": captions.get(name, [""]), "hyp": hyp, **m}
                )
                detailed_results.append({**m, "img_name": name, "hyp": hyp})

    df = pd.DataFrame(results)
    print("\n" + "=" * 30 + " FINAL RESEARCH SCORES " + "=" * 30)
    for m in ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]:
        print(f"{m.upper():10s}: {df[m].mean():.4f}")

    results_summary = {
        m: df[m].mean()
        for m in ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]
    }
    results_summary["samples"] = len(results)
    pd.DataFrame([results_summary]).to_csv(
        os.path.join(OUTPUT_DIR, "test_results.csv"), index=False
    )
    pd.DataFrame(detailed_results).to_csv(
        os.path.join(OUTPUT_DIR, "detailed_results.csv"), index=False
    )

    sample = random.choice(results)
    print(f"\nRANDOM SAMPLE: {sample['img_name']}")
    print(f"Generated: {sample['hyp']}")
    print("Actual:")
    for i, ref in enumerate(sample["ref"], 1):
        print(f"  {i}. {ref}")
    print(
        f"\n BLEU-1:  {sample['bleu1']:.4f}\n BLEU-2:  {sample['bleu2']:.4f}\n BLEU-3:  {sample['bleu3']:.4f}\n BLEU-4:  {sample['bleu4']:.4f}\n METEOR:  {sample['meteor']:.4f}\n ROUGE-L: {sample['rouge_l']:.4f}\n CIDER:   {sample['cider']:.4f}"
    )


if __name__ == "__main__":
    test()
