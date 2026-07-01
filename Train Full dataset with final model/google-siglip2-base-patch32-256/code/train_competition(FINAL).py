import os
import time
import random
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL_FOLDER))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\Python Projects\Train Full dataset with final model\Final_Hybrid_Dataset_60k"
MODEL_DIR = os.path.join(MODEL_FOLDER, "model")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODEL_NAME = "google/siglip2-base-patch32-256"
HF_TOKEN = "YOUR_HF_TOKEN"

BATCH_SIZE = 64               # Fill 12GB VRAM (Fastest burst)
ACCUMULATION_STEPS = 1        # Sharp updates
EPOCHS = 10                   # Trends lock in after 10 epochs
MAX_LR = 1.5e-4               # Fast convergence

USE_AMP = True
NUM_WORKERS = 4
MAX_SEQ_LEN = 32

FEATURE_DIM = 768
EMBED_DIM = 768
NHEAD = 12                    # 64 units/head (Standard)
NUM_LAYERS = 6                # Judge the encoder, save 60% time
HIDDEN_DIM = 2048             # Reduce math overhead
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
MIN_FREQ = 5                  # Small vocab = Faster final layer



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
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in sentence.split()]
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
        if len(tokens) < max_len:
            tokens += [self.word2idx["<PAD>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len - 1] + [self.word2idx["<EOS>"]]
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
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        pixel_values = self.transform(image)

        all_caps = self.captions.get(img_name, [""])
        cap = random.choice(all_caps)
        tokens = self.vocab.encode(cap, MAX_SEQ_LEN)
        return pixel_values, torch.tensor(tokens)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
        
        for param in self.siglip.parameters():
            param.requires_grad = False
             
        self.feature_dim = FEATURE_DIM

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.siglip.vision_model(pixel_values)
        return outputs.last_hidden_state


class Decoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab_size=10000,
                 feature_dim=FEATURE_DIM, num_layers=NUM_LAYERS, nhead=NHEAD, dropout=DROPOUT):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.word2idx = None

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.fc_feature = nn.Linear(feature_dim, embed_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, features, captions):
        batch_size = captions.size(0)
        seq_len = captions.size(1)

        embed = self.embedding(captions) * (self.embed_dim ** 0.5)
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

        generated = torch.full((batch_size, 1), self.word2idx["<SOS>"], dtype=torch.long).to(features.device)

        for _ in range(max_len - 1):
            embed = self.embedding(generated) * (self.embed_dim ** 0.5)
            embed = embed + self.pos_encoder[:, :generated.size(1), :]

            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(features.device)
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
                    embed = self.embedding(tgt) * (self.embed_dim ** 0.5)
                    embed = embed + self.pos_encoder[:, :tgt.size(1), :]

                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
                    mem_slice = memory[b:b + 1]

                    decoder_output = self.decoder(embed, mem_slice, tgt_mask=tgt_mask)
                    logits = self.fc_out(decoder_output[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)

                    top_probs, top_idx = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        candidates.append((seq + [top_idx[0, i].item()], score + top_probs[0, i].item()))

                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                if all(s[-1] == self.word2idx["<EOS>"] for s, sc in beams):
                    break

            final_captions.append(beams[0][0])

        return torch.tensor([seq + [0] * (max_len - len(seq)) for seq in final_captions], dtype=torch.long)


class CaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=vocab_size)

    def forward(self, pixel_values, captions):
        features = self.encoder(pixel_values)
        outputs = self.decoder(features, captions)
        return outputs


def save_training_log(log_file, epoch, train_loss, val_loss, lr, time_taken):
    log_data = {"Epoch": [epoch], "Train Loss": [train_loss], "Val Loss": [val_loss], "Learning Rate": [lr],
                "Time (s)": [time_taken]}

    if os.path.exists(log_file):
        try:
            df_existing = pd.read_excel(log_file)
            df_new = pd.DataFrame(log_data)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(log_file, index=False)
        except:
            os.remove(log_file)
            df_new = pd.DataFrame(log_data)
            df_new.to_excel(log_file, index=False)
    else:
        df_new = pd.DataFrame(log_data)
        df_new.to_excel(log_file, index=False)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    print("Loading data...")
    train_list = load_image_list(os.path.join(DATA_DIR, "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "val.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))

    print(f"Train: {len(train_list)}, Val: {len(val_list)}, Captions: {len(captions)}")

    vocab = Vocabulary(min_freq=MIN_FREQ)
    for img_name, caps in captions.items():
        for cap in caps:
            vocab.add_sentence(cap)

    vocab.build_vocab()
    print(f"Vocab size (min_freq={MIN_FREQ}): {vocab.n_words}")

    print("Loading model...")

    train_dataset = BanglaCaptionDataset(train_list, captions, vocab, os.path.join(DATA_DIR, "images"), is_train=True)
    val_dataset = BanglaCaptionDataset(val_list, captions, vocab, os.path.join(DATA_DIR, "images"), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)

    model = CaptionModel(vocab.n_words).to(DEVICE)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters()]
    all_params = encoder_params + decoder_params

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR,
                                                    steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
                                                    epochs=EPOCHS, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    print(f"\nSigLIP2 MODE: {NUM_LAYERS} Layers | {HIDDEN_DIM} Hidden | DROPOUT: {DROPOUT} | MAX_LR: {MAX_LR}")
    print(f"Vocab Filter: min_freq={MIN_FREQ} | Vision: Fully Frozen")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE * ACCUMULATION_STEPS}\n")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (pixel_values, tokens) in enumerate(pbar):
            pixel_values, tokens = pixel_values.to(DEVICE), tokens.to(DEVICE)

            with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
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

            total_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({"loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for pixel_values, tokens in val_loader:
                pixel_values, tokens = pixel_values.to(DEVICE), tokens.to(DEVICE)
                with torch.amp.autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(pixel_values, tokens)
                    outputs = outputs[:, :-1].contiguous().view(-1, vocab.n_words)
                    targets = tokens[:, 1:].contiguous().view(-1)
                    loss = criterion(outputs, targets)
                val_loss += loss.float().item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Time: {epoch_time:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n")

        save_training_log(log_file, epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_time)

        if (epoch + 1) % 2 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "loss": avg_train_loss, "val_loss": avg_val_loss, "vocab": vocab.word2idx},
                os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt"))

    torch.save({"model_state_dict": model.state_dict(), "vocab": vocab.word2idx},
               os.path.join(OUTPUT_DIR, "final_model.pt"))
    print("Training complete!")


if __name__ == "__main__":
    train()
