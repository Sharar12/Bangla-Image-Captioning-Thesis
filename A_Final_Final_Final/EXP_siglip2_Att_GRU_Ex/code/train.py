import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import time
import random
import logging
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
os.environ["HF_HOME"] = os.path.join(MODEL_FOLDER, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_PATH = os.path.join(MODELS_DIR, "BanglaGPT")
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EPOCHS = 20
MAX_LR = 5e-4
MAX_SEQ_LEN = 26  # fallback; overwritten with p99 at runtime
USE_AMP = True
NUM_WORKERS = 6
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
UNFREEZE_VISION_LAYERS = 4
WEIGHT_DECAY = 0.5

TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05
TRAIN_SEED = 8316

GRU_NUM_LAYERS = 4
LSTM_NUM_LAYERS = 4
UNFREEZE_GPT_LAYERS = 4
BANGLA_GPT_NAME = "shahidul034/BanglaGPT"
BANGLA_GPT_HIDDEN = 768

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class ModelOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.count = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1

    def build_from_captions(self, captions_dict, min_freq=1):
        from collections import Counter

        freq = Counter()
        for caps in captions_dict.values():
            for cap in caps:
                for word in cap.split():
                    freq[word] += 1
        for word, count in freq.items():
            if count >= min_freq:
                self.add_word(word)

    def encode(self, caption, max_len):
        tokens = caption.split()
        ids = (
            [self.word2idx["<start>"]]
            + [self.word2idx.get(w, self.word2idx["<unk>"]) for w in tokens]
            + [self.word2idx["<end>"]]
        )
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.word2idx["<end>"]]
        ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        return torch.tensor(ids[:max_len], dtype=torch.long)

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

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "idx2word": {str(k): v for k, v in self.idx2word.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

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
            BANGLA_GPT_PATH if os.path.exists(BANGLA_GPT_PATH) else BANGLA_GPT_NAME
        )
        self.gpt = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
        for param in self.gpt.parameters():
            param.requires_grad = False

        if UNFREEZE_GPT_LAYERS > 0:
            gpt_layers = getattr(self.gpt, "h", None)
            if gpt_layers is None:
                gpt_layers = getattr(getattr(self.gpt, "transformer", None), "h", None)
            if gpt_layers is not None and hasattr(gpt_layers, "__len__"):
                total = len(gpt_layers)
                start = max(0, total - UNFREEZE_GPT_LAYERS)
                for i in range(start, total):
                    for p in gpt_layers[i].parameters():
                        p.requires_grad = True
                print(
                    f"BanglaGPT: Unfreezing last {UNFREEZE_GPT_LAYERS}/{total} layers"
                )

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
    def __init__(self, unfreeze_layers=UNFREEZE_VISION_LAYERS):
        super().__init__()
        model_path = (
            SIGLIP_MODEL_PATH
            if os.path.exists(SIGLIP_MODEL_PATH)
            else "google/siglip2-base-patch32-256"
        )
        self.siglip = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.unfreeze_layers = unfreeze_layers
        self._fully_frozen = True
        self._configure_gradients()

    def _configure_gradients(self):
        if self.unfreeze_layers == 0:
            for param in self.siglip.parameters():
                param.requires_grad = True
            self._fully_frozen = False
        elif self.unfreeze_layers > 0:
            for param in self.siglip.parameters():
                param.requires_grad = False
            layers = self.siglip.vision_model.encoder.layers
            start = max(0, len(layers) - self.unfreeze_layers)
            for i in range(start, len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
            self._fully_frozen = False
        else:
            for param in self.siglip.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        if self._fully_frozen:
            with torch.no_grad():
                return self.siglip.vision_model(pixel_values).last_hidden_state
        return self.siglip.vision_model(pixel_values).last_hidden_state


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, unfreeze_vision=UNFREEZE_VISION_LAYERS):
        super().__init__()
        self.encoder = Encoder(unfreeze_layers=unfreeze_vision)
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = MultiDecoderFusion(vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, label_smoothing=LABEL_SMOOTHING
        )

    def forward(self, pixel_values, input_ids, labels=None):
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        logits = self.decoder(visual_tokens, input_ids)
        if labels is None:
            labels = input_ids.clone()
            labels[labels == 0] = -100
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = self.loss_fn(
            shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1)
        )
        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_captions(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        start_id, end_id = vocab.word2idx["<start>"], vocab.word2idx["<end>"]
        generated = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=device
        )

        unk_id = vocab.word2idx.get("<unk>", 3)
        for _ in range(max_new_tokens - 1):
            logits = self.decoder(visual_tokens, generated)
            logits[:, -1, unk_id] = -1e9
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return [vocab.decode(seq.tolist()) for seq in generated]

    def generate_caption(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        return self.generate_captions(pixel_values, vocab, max_new_tokens)[0]

    @torch.no_grad()
    def generate_beam(
        self, pixel_values, vocab, beam_size=5, max_new_tokens=MAX_SEQ_LEN
    ):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        start_id = vocab.word2idx["<start>"]
        end_id = vocab.word2idx["<end>"]

        sequences = [[start_id]]
        scores = [0.0]

        for _ in range(max_new_tokens):
            all_candidates = []
            for i, seq in enumerate(sequences):
                if seq[-1] == end_id:
                    all_candidates.append((scores[i], seq))
                    continue
                ids = torch.tensor([seq], device=pixel_values.device)
                logits = self.decoder(visual_tokens, ids)[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)[0]
                top = log_probs.topk(beam_size)
                for lp, tok in zip(top.values, top.indices):
                    all_candidates.append((scores[i] + lp.item(), seq + [tok.item()]))

            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = [c[1] for c in ordered[:beam_size]]
            scores = [c[0] for c in ordered[:beam_size]]

            if all(s[-1] == end_id for s in sequences):
                break

        return vocab.decode(sequences[0])


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
    def __init__(
        self, image_list, captions, vocab, image_dir, processor_path, is_train=True
    ):
        self.image_dir = image_dir
        self.vocab = vocab
        self.max_len = MAX_SEQ_LEN
        self.is_train = is_train
        self.processor = AutoImageProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
        self.aug = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1
                    ),
                ]
            )
            if is_train
            else None
        )
        self.pairs = []
        for img_name in image_list:
            caps = captions.get(img_name, [""])
            if is_train:
                for cap in caps:
                    self.pairs.append((img_name, cap))
            else:
                for cap in caps:
                    self.pairs.append((img_name, cap))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, cap = self.pairs[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
            if self.aug:
                image = self.aug(image)
            pixel_values = self.processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 256, 256)
        caption_ids = self.vocab.encode(cap, self.max_len)
        return pixel_values, caption_ids


def save_training_log(log_file, epoch, train_loss, val_loss, lr, time_taken):
    df_new = pd.DataFrame(
        {
            "Epoch": [epoch],
            "Train Loss": [train_loss],
            "Val Loss": [val_loss],
            "Learning Rate": [f"{lr:.10f}"],
            "Time (s)": [time_taken],
        }
    )
    fallback = log_file.replace(".xlsx", ".csv")
    try:
        if os.path.exists(log_file):
            pd.concat([pd.read_excel(log_file), df_new], ignore_index=True).to_excel(
                log_file, index=False
            )
        else:
            df_new.to_excel(log_file, index=False)
    except Exception:
        if os.path.exists(fallback):
            pd.concat([pd.read_csv(fallback), df_new], ignore_index=True).to_csv(
                fallback, index=False
            )
        else:
            df_new.to_csv(fallback, index=False)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(
        OUTPUT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    print("Loading data...")
    captions = load_captions(os.path.join(DATA_DIR, "caption.txt"))
    image_names = list(captions.keys())
    n = len(image_names)

    generator = torch.Generator().manual_seed(TRAIN_SEED)
    train_idx, val_idx, test_idx = [], [], []
    perm = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_list = [image_names[i] for i in train_idx]
    val_list = [image_names[i] for i in val_idx]
    test_list = [image_names[i] for i in test_idx]

    print(f"\n{'=' * 50}")
    for name, lst in [("Train", train_list), ("Val", val_list), ("Test", test_list)]:
        print(f"{name:<12} {len(lst):<10} {len(lst) / n * 100:.2f}%")
    print(f"{'Total':<12} {n:<10} {'100.00%'}")
    print(f"{'=' * 50}\nImages: {len(captions)}")

    all_lengths = [len(cap.split()) + 2 for caps in captions.values() for cap in caps]
    all_lengths.sort()
    nl = len(all_lengths)
    p95 = all_lengths[int(nl * 0.95)]
    p99 = all_lengths[int(nl * 0.99)]
    print(
        f"Caption lengths — mean: {sum(all_lengths) / nl:.1f}, "
        f"p90: {all_lengths[int(nl * 0.9)]}, "
        f"p95: {p95}, "
        f"p99: {p99}, "
        f"max: {all_lengths[-1]}"
    )
    MAX_SEQ_LEN = p99
    print(f"Setting MAX_SEQ_LEN = {MAX_SEQ_LEN}")

    print("Building word-level vocabulary from training captions...")
    train_captions = {k: captions[k] for k in train_list}
    vocab = Vocabulary()
    vocab.build_from_captions(train_captions, min_freq=1)
    print(f"Vocabulary size: {vocab.vocab_size}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    vocab.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    print("Creating datasets...")
    siglip_path = (
        SIGLIP_MODEL_PATH
        if os.path.exists(SIGLIP_MODEL_PATH)
        else "google/siglip2-base-patch32-256"
    )
    train_dataset = BanglaCaptionDataset(
        train_list,
        captions,
        vocab,
        os.path.join(DATA_DIR, "images"),
        siglip_path,
        is_train=True,
    )
    val_dataset = BanglaCaptionDataset(
        val_list,
        captions,
        vocab,
        os.path.join(DATA_DIR, "images"),
        siglip_path,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    model = CaptionModel(vocab_size=vocab.vocab_size).to(DEVICE)

    if torch.__version__.startswith("2"):
        model = torch.compile(model)

    vision_params = [p for p in model.encoder.parameters() if p.requires_grad]
    proj_params = list(model.vision_projection.parameters())
    decoder_params = list(model.decoder.parameters())
    all_params = vision_params + proj_params + decoder_params
    total_trainable = sum(p.numel() for p in all_params)

    optimizer = torch.optim.AdamW(
        [
            {"params": vision_params, "lr": 5e-5},
            {"params": proj_params, "lr": 2e-4},
            {"params": decoder_params, "lr": 3e-4},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    total_steps = (
        (len(train_loader) + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS * EPOCHS
    )
    warmup_steps = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n{'=' * 50}")
    print(f"Architecture: SigLIP2 → Linear(768,{HIDDEN_DIM}) → Sequential Pipeline")
    print(f"  ├─ GRU ({GRU_NUM_LAYERS}-layer + attention) → processes text+image")
    print(f"  ├─ LSTM ({LSTM_NUM_LAYERS}-layer + attention) → processes GRU output")
    print(f"  └─ BanglaGPT → generates caption from LSTM features")
    vision_status = (
        f"Last {UNFREEZE_VISION_LAYERS} layers"
        if UNFREEZE_VISION_LAYERS > 0
        else "All"
        if UNFREEZE_VISION_LAYERS == 0
        else "Frozen"
    )
    gpt_status = (
        f"Last {UNFREEZE_GPT_LAYERS} layers" if UNFREEZE_GPT_LAYERS > 0 else "Frozen"
    )
    print(f"Vision: {vision_status} | BanglaGPT: {gpt_status}")
    print(
        f"Trainable: {total_trainable:,} | Vocab: {vocab.vocab_size:,} | MaxLen: {MAX_SEQ_LEN}"
    )
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"{'=' * 50}\n")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        running_loss = 0.0
        running_count = 0

        for batch_idx, (pixel_values, caption_ids) in enumerate(pbar):
            pixel_values = pixel_values.to(DEVICE)
            caption_ids = caption_ids.to(DEVICE)
            labels = caption_ids.clone()
            labels[labels == 0] = -100

            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(pixel_values, caption_ids, labels=labels)
                loss = outputs.loss / ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            batch_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += batch_loss
            running_loss += batch_loss
            running_count += 1
            pbar.set_postfix({"loss": f"{running_loss / running_count:.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for pixel_values, caption_ids in val_loader:
                pixel_values = pixel_values.to(DEVICE)
                caption_ids = caption_ids.to(DEVICE)
                labels = caption_ids.clone()
                labels[labels == 0] = -100
                with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                    outputs = model(pixel_values, caption_ids, labels=labels)
                val_loss += outputs.loss.float().item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[2]["lr"]

        sample_img, sample_cap = val_dataset[0]
        with torch.no_grad():
            sample_pv = sample_img.unsqueeze(0).to(DEVICE)
            pred = model.generate_caption(sample_pv, vocab)
        ref = vocab.decode(sample_cap.tolist())
        print(f"  REF : {ref}")
        print(f"  PRED: {pred}")

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | {epoch_time:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}\n"
        )
        save_training_log(
            log_file, epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_time
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                os.path.join(OUTPUT_DIR, "best_model.pt"),
            )
            print(f"  New best model saved (val_loss: {best_val_loss:.4f})")

        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            ckpt_path,
        )
        prev = epoch + 1 - 3
        if prev > 0:
            old = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{prev}.pt")
            if os.path.exists(old):
                os.remove(old)

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(OUTPUT_DIR, "final_model.pt"),
    )
    print("Training complete!")


if __name__ == "__main__":
    train()
