import os
import sys

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import random
import warnings
import logging
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*Enum subclasses.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\Python Projects\Research Papers Test\BNNATURE"
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")

BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EPOCHS = 20
MAX_LR = 5e-4
MAX_SEQ_LEN = 51

USE_AMP = True
NUM_WORKERS = 8

FEATURE_DIM = 2048
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
MIN_FREQ = 0
UNFREEZE_VISION_LAYERS = 8


class Vocabulary:
    def __init__(self, min_freq=MIN_FREQ):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4
        self.word_counts = defaultdict(int)
        self.min_freq = min_freq

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.word_counts[word] += 1

    def build_vocab(self):
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1

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
            if w != "<SOS>":
                words.append(w)
        return " ".join(words)


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
    def __init__(self, image_list, captions, vocab, image_dir, is_train=True):
        self.image_list = image_list
        self.captions = captions
        self.vocab = vocab
        self.image_dir = image_dir
        self.max_len = MAX_SEQ_LEN

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((299, 299)), transforms.ToTensor(), normalize]
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros(3, 299, 299)
        all_caps = self.captions.get(img_name, [""])
        cap = random.choice(all_caps)
        tokens = self.vocab.encode(cap, self.max_len)
        return pixel_values, torch.tensor(tokens)


class Encoder(nn.Module):
    def __init__(self, unfreeze_layers=UNFREEZE_VISION_LAYERS):
        super().__init__()
        self.inception = models.inception_v3(weights="DEFAULT")
        self.inception.fc = nn.Identity()
        self.inception.AuxLogits = nn.Identity()
        self.inception.avgpool = nn.Identity()
        self.feature_dim = FEATURE_DIM
        self.unfreeze_layers = unfreeze_layers
        self._fully_frozen = True
        self._configure_gradients()
        self.inception.Mixed_7c.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        self._features = output

    def _configure_gradients(self):
        if self.unfreeze_layers == 0:
            print("Encoder: Unfreezing 100% of InceptionV3.")
            for param in self.inception.parameters():
                param.requires_grad = True
            self._fully_frozen = False
        elif self.unfreeze_layers > 0:
            print(
                f"Encoder: Unfreezing the last {self.unfreeze_layers} modules of InceptionV3."
            )
            for param in self.inception.parameters():
                param.requires_grad = False
            children = list(self.inception.named_children())
            feature_modules = [
                (n, m) for n, m in children if n not in ("fc", "AuxLogits", "avgpool")
            ]
            total = len(feature_modules)
            start = max(0, total - self.unfreeze_layers)
            for i in range(start, total):
                for param in feature_modules[i][1].parameters():
                    param.requires_grad = True
            self._fully_frozen = False
        else:
            print("Encoder: InceptionV3 is 100% frozen.")
            for param in self.inception.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        if self._fully_frozen:
            with torch.no_grad():
                self.inception(pixel_values)
        else:
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
    def __init__(self, vocab_size, unfreeze_layers=UNFREEZE_VISION_LAYERS):
        super().__init__()
        self.encoder = Encoder(unfreeze_layers=unfreeze_layers)
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = BanglaLuongAttentionDecoder(vocab_size=vocab_size)

    def forward(self, pixel_values, captions):
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        outputs = self.decoder(visual_tokens, captions)
        return outputs


def save_training_log(log_file, epoch, train_loss, val_loss, lr, time_taken):
    log_data = {
        "Epoch": [epoch],
        "Train Loss": [train_loss],
        "Val Loss": [val_loss],
        "Learning Rate": [lr],
        "Time (s)": [time_taken],
    }
    df_new = pd.DataFrame(log_data)
    fallback = log_file.replace(".xlsx", ".csv")

    try:
        if os.path.exists(log_file):
            df_existing = pd.read_excel(log_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(log_file, index=False)
        else:
            df_new.to_excel(log_file, index=False)
    except Exception:
        if os.path.exists(fallback):
            df_existing = pd.read_csv(fallback)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(fallback, index=False)
        else:
            df_new.to_csv(fallback, index=False)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(
        OUTPUT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    print("Loading data...")
    train_list = load_image_list(os.path.join(DATA_DIR, "caption", "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "caption", "validation.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))

    total = len(train_list) + len(val_list)
    print(f"\n{'=' * 50}")
    print(f"{'Split':<12} {'Count':<10} {'Percentage':<10}")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"{'Train':<12} {len(train_list):<10} {len(train_list) / total * 100:.2f}%")
    print(f"{'Val':<12} {len(val_list):<10} {len(val_list) / total * 100:.2f}%")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"{'Total':<12} {total:<10} {'100.00%'}")
    print(f"{'=' * 50}\n")

    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=MIN_FREQ)
    for caps in captions.values():
        for cap in caps:
            vocab.add_sentence(cap)
    vocab.build_vocab()
    print(f"Vocabulary size: {vocab.n_words}")

    train_dataset = BanglaCaptionDataset(
        train_list, captions, vocab, os.path.join(DATA_DIR, "images"), is_train=True
    )
    val_dataset = BanglaCaptionDataset(
        val_list, captions, vocab, os.path.join(DATA_DIR, "images"), is_train=False
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

    model = CaptionModel(vocab.n_words, unfreeze_layers=UNFREEZE_VISION_LAYERS).to(
        DEVICE
    )
    model.decoder.word2idx = vocab.word2idx

    all_params = [p for p in model.parameters() if p.requires_grad]

    total_trainable = sum(p.numel() for p in all_params)
    trainable_encoder = sum(
        p.requires_grad for p in model.encoder.inception.parameters()
    )
    encoder_status = (
        "Fully Frozen"
        if UNFREEZE_VISION_LAYERS < 0
        else (
            "All Trainable"
            if UNFREEZE_VISION_LAYERS == 0
            else f"Last {UNFREEZE_VISION_LAYERS} Modules Unfrozen ({trainable_encoder} params)"
        )
    )
    print(f"\nArchitecture: InceptionV3 Vision -> GRU Decoder")
    print(f"Encoder: {encoder_status}")
    print(f"Trainable params: {total_trainable:,}")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE * ACCUMULATION_STEPS}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=max(1, len(train_loader) // ACCUMULATION_STEPS),
        epochs=EPOCHS,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        running_loss = 0.0
        running_count = 0
        for batch_idx, (pixel_values, tokens) in enumerate(pbar):
            pixel_values, tokens = pixel_values.to(DEVICE), tokens.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(pixel_values, tokens)
                outputs = outputs[:, :-1].contiguous().view(-1, vocab.n_words)
                targets = tokens[:, 1:].contiguous().view(-1)
                loss = criterion(outputs, targets) / ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            batch_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += batch_loss
            running_loss += batch_loss
            running_count += 1
            avg_loss = running_loss / running_count
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for pixel_values, tokens in val_loader:
                pixel_values, tokens = pixel_values.to(DEVICE), tokens.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                    outputs = model(pixel_values, tokens)
                    outputs = outputs[:, :-1].contiguous().view(-1, vocab.n_words)
                    targets = tokens[:, 1:].contiguous().view(-1)
                    loss = criterion(outputs, targets)
                val_loss += loss.float().item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Time: {epoch_time:.1f}s | "
            f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n"
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
                    "word2idx": vocab.word2idx,
                },
                os.path.join(OUTPUT_DIR, "best_model.pt"),
            )
            print(f"  New best model saved (val_loss: {best_val_loss:.4f})")

        if (epoch + 1) == EPOCHS:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "word2idx": vocab.word2idx,
                },
                os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word2idx": vocab.word2idx,
        },
        os.path.join(OUTPUT_DIR, "final_model.pt"),
    )
    print("Training complete!")


if __name__ == "__main__":
    train()
