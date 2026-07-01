import os
import sys
import time
import random
import warnings
import logging
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
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

MODEL_DIR = os.path.join(MODEL_FOLDER, "model")
MODEL_NAME = "google/siglip2-base-patch32-256"
HF_TOKEN = "YOUR_HF_TOKEN"

# --- NEW SETTING FOR ENCODER UNFREEZING ---
# -1 = Fully Frozen (Original behavior)
#  0 = Fully Unfrozen (Train all layers of SigLIP2)
#  N = Unfreeze only the last N transformer layers (e.g., 2 or 4)
UNFREEZE_LAYERS = 8

BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EPOCHS = 20
MAX_LR = 5e-4

USE_AMP = True
NUM_WORKERS = 8
MAX_SEQ_LEN = 51

FEATURE_DIM = 768
EMBED_DIM = 768
NHEAD = 12
NUM_LAYERS = 6
HIDDEN_DIM = 2048
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
MIN_FREQ = 0


class Vocabulary:
    def __init__(self, min_freq=3):
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
                captions[parts[0]].append(parts[1])
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
    def __init__(self, image_list, captions, vocab, image_dir, is_train=True):
        self.image_list = image_list
        self.captions = captions
        self.vocab = vocab
        self.image_dir = image_dir
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize((256, 256)), transforms.ToTensor(), normalize]
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros(3, 256, 256)
        all_caps = self.captions.get(img_name, [""])
        cap = random.choice(all_caps)
        tokens = self.vocab.encode(cap, MAX_SEQ_LEN)
        return pixel_values, torch.tensor(tokens)


class Encoder(nn.Module):
    def __init__(self, unfreeze_layers=-1):
        super().__init__()
        # Load from normalized string path or repo
        model_path = (
            os.path.abspath(MODEL_DIR) if os.path.exists(MODEL_DIR) else MODEL_DIR
        )
        self.siglip = AutoModel.from_pretrained(
            model_path, token=HF_TOKEN, trust_remote_code=True
        )
        self.feature_dim = FEATURE_DIM
        self.unfreeze_layers = unfreeze_layers

        # Configure Grad rules based on user settings
        self._configure_gradients()

    def _configure_gradients(self):
        if self.unfreeze_layers == 0:
            print("Encoder Configuration: Unfreezing 100% of the Vision Encoder.")
            for param in self.siglip.parameters():
                param.requires_grad = True
        elif self.unfreeze_layers > 0:
            print(
                f"Encoder Configuration: Unfreezing the last {self.unfreeze_layers} layers of Vision Transformer."
            )
            # First freeze everything
            for param in self.siglip.parameters():
                param.requires_grad = False

            # Target the transformer blocks inside the Hugging Face Siglip Vision Model structure
            layers = self.siglip.vision_model.encoder.layers
            total_layers = len(layers)
            start_index = max(0, total_layers - self.unfreeze_layers)

            for i in range(start_index, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
        else:
            print("Encoder Configuration: Vision Encoder is 100% frozen.")
            for param in self.siglip.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        # REMOVED the hard torch.no_grad block so gradients can scale when unfrozen
        outputs = self.siglip.vision_model(pixel_values)
        return outputs.last_hidden_state


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=10000,
        feature_dim=FEATURE_DIM,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        dropout=DROPOUT,
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
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, features, captions):
        seq_len = captions.size(1)
        embed = self.embedding(captions) * (self.embed_dim**0.5)
        embed = embed + self.pos_encoder[:, :seq_len, :]
        memory = self.fc_feature(features)
        memory = self.encoder(memory)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        decoder_output = self.decoder(embed, memory, tgt_mask=tgt_mask)
        return self.fc_out(decoder_output)

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


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, unfreeze_layers=-1):
        super().__init__()
        self.encoder = Encoder(unfreeze_layers=unfreeze_layers)
        self.decoder = Decoder(vocab_size=vocab_size)

    def forward(self, pixel_values, captions):
        features = self.encoder(pixel_values)
        outputs = self.decoder(features, captions)
        return outputs


def save_training_log(log_file, epoch, train_loss, val_loss, lr, time_taken):
    log_data = {
        "Epoch": [epoch],
        "Train Loss": [train_loss],
        "Val Loss": [val_loss],
        "Learning Rate": [lr],
        "Time (s)": [time_taken],
    }
    if os.path.exists(log_file):
        try:
            df_existing = pd.read_excel(log_file)
            df_combined = pd.concat(
                [df_existing, pd.DataFrame(log_data)], ignore_index=True
            )
            df_combined.to_excel(log_file, index=False)
        except:
            os.remove(log_file)
            pd.DataFrame(log_data).to_excel(log_file, index=False)
    else:
        pd.DataFrame(log_data).to_excel(log_file, index=False)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(
        OUTPUT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    print("Loading data configuration paths...")
    train_list = load_image_list(os.path.join(DATA_DIR, "caption", "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "caption", "validation.txt"))
    test_list = load_image_list(os.path.join(DATA_DIR, "caption", "test.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))

    total = len(train_list) + len(val_list) + len(test_list)
    print(f"\n{'=' * 50}")
    print(f"{'Split':<12} {'Count':<10} {'Percentage':<10}")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"{'Train':<12} {len(train_list):<10} {len(train_list) / total * 100:.2f}%")
    print(f"{'Val':<12} {len(val_list):<10} {len(val_list) / total * 100:.2f}%")
    print(f"{'Test':<12} {len(test_list):<10} {len(test_list) / total * 100:.2f}%")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"{'Total':<12} {total:<10} {'100.00%'}")
    print(f"{'=' * 50}\n")

    vocab = Vocabulary(min_freq=MIN_FREQ)
    for img_name, caps in captions.items():
        for cap in caps:
            vocab.add_sentence(cap)
    vocab.build_vocab()

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

    print(f"Loading Base Encoder from: {MODEL_DIR}")
    # Passes unfreeze layer choice down to execution steps
    model = CaptionModel(vocab.n_words, unfreeze_layers=UNFREEZE_LAYERS).to(DEVICE)
    model.decoder.word2idx = vocab.word2idx

    # Dynamically captures encoder parameters if unfrozen
    all_params = [p for p in model.parameters() if p.requires_grad]

    # Quick sanity validation print
    print(f"Total Parameter Groups Tagged for Optimization: {len(all_params)}")

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
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

            current_batch_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += current_batch_loss

            # 👇 LIVE LOSS TERMINAL VIEW FEED
            pbar.set_postfix({"loss": f"{current_batch_loss:.4f}"})

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
            f"\nSummary Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
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
                    "vocab": vocab.word2idx,
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
                    "vocab": vocab.word2idx,
                },
                os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab.word2idx,
        },
        os.path.join(OUTPUT_DIR, "final_model.pt"),
    )
    print("Training complete!")


if __name__ == "__main__":
    train()
