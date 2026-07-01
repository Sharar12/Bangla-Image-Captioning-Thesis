import os
import time
import random
import warnings
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoImageProcessor,
    BertLMHeadModel,
    AutoTokenizer,
    BertConfig,
)
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd

warnings.filterwarnings("ignore", message=".*register_constant.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\Python Projects\Research Papers Test\BNNATURE"
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_BERT_PATH = os.path.join(MODELS_DIR, "banglabert")
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EXTRA_EPOCHS = 5
MAX_LR = 2e-5
MAX_SEQ_LEN = 51

USE_AMP = True
NUM_WORKERS = 8

FEATURE_DIM = 768
HIDDEN_SIZE = 768
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
UNFREEZE_VISION_LAYERS = 8
UNFREEZE_BANGLA_LAYERS = 8


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
    def __init__(self, image_list, captions, tokenizer, image_dir, is_train=True):
        self.image_list = image_list
        self.captions = captions
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.max_len = MAX_SEQ_LEN

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
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        pixel_values = self.transform(image)

        all_caps = self.captions.get(img_name, [""])
        cap = random.choice(all_caps)

        tokens = self.tokenizer(
            cap,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            pixel_values,
            tokens["input_ids"].squeeze(0),
            tokens["attention_mask"].squeeze(0),
        )


@dataclass
class ModelOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, unfreeze_layers=UNFREEZE_VISION_LAYERS):
        super().__init__()
        self.siglip = None
        self.feature_dim = FEATURE_DIM

    def forward(self, pixel_values):
        return self.siglip.vision_model(pixel_values).last_hidden_state


class CaptionModel(nn.Module):
    def __init__(self, checkpoint=None):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
        )
        self.banglabert = None
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, label_smoothing=LABEL_SMOOTHING
        )

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        text_embeds = self.banglabert.get_input_embeddings()(input_ids)
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

        if labels is None:
            labels = input_ids.clone()
            text_attention_mask = attention_mask[:, num_vis:]
            labels[text_attention_mask == 0] = -100
        full_labels = torch.full(
            (batch_size, num_vis + seq_len),
            -100,
            device=input_ids.device,
            dtype=torch.long,
        )
        full_labels[:, num_vis:] = labels

        outputs = self.banglabert(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_labels[..., 1:].contiguous()
        loss = self.loss_fn(
            shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1)
        )

        return ModelOutput(loss=loss, logits=logits)


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


def resume_train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(
        OUTPUT_DIR, f"training_continue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_20.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    print("Loading data...")
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))
    train_list = load_image_list(os.path.join(DATA_DIR, "caption", "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "caption", "validation.txt"))

    print("Loading BanglaBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BANGLA_BERT_PATH
        if os.path.exists(BANGLA_BERT_PATH)
        else "csebuetnlp/banglabert",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating datasets...")
    train_dataset = BanglaCaptionDataset(
        train_list, captions, tokenizer, os.path.join(DATA_DIR, "images"), is_train=True
    )
    val_dataset = BanglaCaptionDataset(
        val_list, captions, tokenizer, os.path.join(DATA_DIR, "images"), is_train=False
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

    print("Building model...")
    model = CaptionModel()

    siglip_path = (
        SIGLIP_MODEL_PATH
        if os.path.exists(SIGLIP_MODEL_PATH)
        else "google/siglip2-base-patch32-256"
    )
    siglip = AutoModel.from_pretrained(siglip_path, trust_remote_code=True)
    setattr(model.encoder, "siglip", siglip)

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
    setattr(model, "banglabert", banglabert)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(DEVICE)

    for param in model.encoder.siglip.parameters():
        param.requires_grad = False
    layers = model.encoder.siglip.vision_model.encoder.layers
    start_idx = max(0, len(layers) - UNFREEZE_VISION_LAYERS)
    for i in range(start_idx, len(layers)):
        for param in layers[i].parameters():
            param.requires_grad = True

    for param in model.banglabert.parameters():
        param.requires_grad = False
    layers = model.banglabert.bert.encoder.layer
    num_to_unfreeze = min(UNFREEZE_BANGLA_LAYERS, len(layers))
    for i in range(len(layers) - num_to_unfreeze, len(layers)):
        for param in layers[i].parameters():
            param.requires_grad = True
    model.banglabert.cls.predictions.decoder.requires_grad_(True)

    vision_params = [
        p for p in model.encoder.parameters() if p.requires_grad and p is not None
    ]
    projection_params = list(model.vision_projection.parameters())
    bangla_params = [p for p in model.banglabert.parameters() if p.requires_grad]
    all_params = vision_params + projection_params + bangla_params

    optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EXTRA_EPOCHS * len(train_loader) // ACCUMULATION_STEPS
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    print(
        f"\nResuming for {EXTRA_EPOCHS} extra epochs with CORRECTED loss (PAD masked)"
    )
    print(f"Trainable params: {sum(p.numel() for p in all_params):,}")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE * ACCUMULATION_STEPS}\n")

    for epoch in range(EXTRA_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Extra Epoch {epoch + 1}/{EXTRA_EPOCHS}")

        running_loss = 0.0
        running_count = 0
        for batch_idx, (pixel_values, input_ids, attention_mask) in enumerate(pbar):
            pixel_values = pixel_values.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(pixel_values, input_ids, attention_mask=attention_mask)
                loss = outputs.loss / ACCUMULATION_STEPS

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
            for pixel_values, input_ids, attention_mask in val_loader:
                pixel_values = pixel_values.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                    outputs = model(
                        pixel_values, input_ids, attention_mask=attention_mask
                    )
                    loss = outputs.loss
                val_loss += loss.float().item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0] if scheduler else MAX_LR

        print(
            f"Extra Epoch {epoch + 1}/{EXTRA_EPOCHS} | Time: {epoch_time:.1f}s | "
            f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n"
        )

        save_training_log(
            log_file, epoch + 1, avg_train_loss, avg_val_loss, current_lr, epoch_time
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            os.path.join(OUTPUT_DIR, f"checkpoint_fixed_epoch_{epoch + 1}.pt"),
        )

    print("Continue training complete!")


if __name__ == "__main__":
    resume_train()
