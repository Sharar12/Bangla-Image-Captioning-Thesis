import os
import json
import time
import random
import warnings
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd

warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\Python Projects\Train Full dataset with final model - For Thesis\Final_Hybrid_Dataset_60k"
PARENT_DIR = os.path.dirname(MODEL_FOLDER)
BANGLABERT_DIR = os.path.join(PARENT_DIR, "siglip2_banglabert")
MODELS_DIR = os.path.join(BANGLABERT_DIR, "models")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_BERT_PATH = os.path.join(MODELS_DIR, "banglabert")
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EPOCHS = 30
MAX_LR = 5e-4
MAX_SEQ_LEN = 31
USE_AMP = True
NUM_WORKERS = 8
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.3
UNFREEZE_VISION_LAYERS = 8


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

    def build_from_captions(self, captions_dict):
        for caps in captions_dict.values():
            for cap in caps:
                for word in cap.split():
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
        self.decoder = BanglaLuongAttentionDecoder(vocab_size=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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
    def __init__(
        self, image_list, captions, vocab, image_dir, processor_path, is_train=True
    ):
        self.image_list = image_list
        self.captions = captions
        self.vocab = vocab
        self.image_dir = image_dir
        self.max_len = MAX_SEQ_LEN
        self.is_train = is_train
        self.processor = AutoImageProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        cap = random.choice(self.captions.get(img_name, [""]))
        caption_ids = self.vocab.encode(cap, self.max_len)
        return pixel_values, caption_ids


def save_training_log(log_file, epoch, train_loss, val_loss, lr, time_taken):
    df_new = pd.DataFrame(
        {
            "Epoch": [epoch],
            "Train Loss": [train_loss],
            "Val Loss": [val_loss],
            "Learning Rate": [lr],
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
    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))
    train_list = load_image_list(os.path.join(DATA_DIR, "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "val.txt"))

    total = len(train_list) + len(val_list)
    print(f"\n{'=' * 50}")
    for name, lst in [("Train", train_list), ("Val", val_list)]:
        print(f"{name:<12} {len(lst):<10} {len(lst) / total * 100:.2f}%")
    print(f"{'Total':<12} {total:<10} {'100.00%'}")
    print(f"{'=' * 50}\nCaptions: {len(captions)}")

    print("Building word-level vocabulary from dataset...")
    vocab = Vocabulary()
    vocab.build_from_captions(captions)
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

    vision_params = [p for p in model.encoder.parameters() if p.requires_grad]
    proj_params = list(model.vision_projection.parameters())
    decoder_params = list(model.decoder.parameters())
    all_params = vision_params + proj_params + decoder_params
    total_trainable = sum(p.numel() for p in all_params)

    optimizer = torch.optim.AdamW(
        [
            {"params": vision_params, "lr": MAX_LR * 0.02},
            {"params": proj_params, "lr": MAX_LR * 0.1},
            {"params": decoder_params, "lr": MAX_LR},
        ],
        weight_decay=0.05,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[MAX_LR * 0.02, MAX_LR * 0.1, MAX_LR],
        steps_per_epoch=max(1, len(train_loader) // ACCUMULATION_STEPS),
        epochs=EPOCHS,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    print(f"\n{'=' * 50}")
    print(f"Architecture: SigLIP2 → Linear(768,{HIDDEN_DIM}) → Luong Attention + GRU")
    print(
        f"Decoder: Embed({EMBED_DIM}) → GRU({HIDDEN_DIM}) → Concat(Attn) → Tanh → Softmax"
    )
    print(
        f"Vision: {'Last ' + str(UNFREEZE_VISION_LAYERS) + ' layers' if UNFREEZE_VISION_LAYERS > 0 else 'All' if UNFREEZE_VISION_LAYERS == 0 else 'Frozen'}"
    )
    print(
        f"Trainable: {total_trainable:,} | Vocab: {vocab.vocab_size:,} | MaxLen: {MAX_SEQ_LEN}"
    )
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"{'=' * 50}\n")

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
        current_lr = scheduler.get_last_lr()[2]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | {epoch_time:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n"
        )
        save_training_log(
            log_file, epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_time
        )

        if (epoch + 1) % 2 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(OUTPUT_DIR, "final_model.pt"),
    )
    print("Training complete!")


if __name__ == "__main__":
    train()
