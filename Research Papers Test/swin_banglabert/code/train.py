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
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
SWIN_MODEL_PATH = os.path.join(MODELS_DIR, "swin-tiny-patch4-window7-224")
BANGLA_BERT_PATH = os.path.join(MODELS_DIR, "banglabert")
BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EPOCHS = 20
MAX_LR = 5e-4
MAX_SEQ_LEN = 51

USE_AMP = True
NUM_WORKERS = 8

FEATURE_DIM = 768
HIDDEN_SIZE = 768
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
UNFREEZE_VISION_LAYERS = 8
UNFREEZE_BANGLA_LAYERS = 8


def download_models():
    os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(SWIN_MODEL_PATH):
        print("Downloading Swin-tiny-patch4-window7-224...")
        model = AutoModel.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224",
            trust_remote_code=True,
        )
        model.save_pretrained(SWIN_MODEL_PATH)
        processor = AutoImageProcessor.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        processor.save_pretrained(SWIN_MODEL_PATH)
        print(f"Swin saved to {SWIN_MODEL_PATH}")
    else:
        print(f"Swin already exists at {SWIN_MODEL_PATH}")

    if not os.path.exists(BANGLA_BERT_PATH):
        print("Downloading BanglaBERT (csebuetnlp/banglabert)...")
        tokenizer = AutoTokenizer.from_pretrained(
            "csebuetnlp/banglabert", trust_remote_code=True
        )
        config = BertConfig.from_pretrained(
            "csebuetnlp/banglabert", trust_remote_code=True
        )
        config.is_decoder = True
        model = BertLMHeadModel.from_pretrained(
            "csebuetnlp/banglabert",
            config=config,
            trust_remote_code=True,
        )
        tokenizer.save_pretrained(BANGLA_BERT_PATH)
        model.save_pretrained(BANGLA_BERT_PATH)
        print(f"BanglaBERT saved to {BANGLA_BERT_PATH}")
    else:
        print(f"BanglaBERT already exists at {BANGLA_BERT_PATH}")

    return SWIN_MODEL_PATH, BANGLA_BERT_PATH


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

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
            pixel_values = self.transform(image)
        except Exception:
            pixel_values = torch.zeros(3, 224, 224)

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
        model_path = (
            SWIN_MODEL_PATH
            if os.path.exists(SWIN_MODEL_PATH)
            else "microsoft/swin-tiny-patch4-window7-224"
        )
        self.swin = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.unfreeze_layers = unfreeze_layers
        self._fully_frozen = True
        self._configure_gradients()

    def _configure_gradients(self):
        if self.unfreeze_layers == 0:
            print("Encoder: Unfreezing 100% of the Vision Encoder.")
            for param in self.swin.parameters():
                param.requires_grad = True
            self._fully_frozen = False
        elif self.unfreeze_layers > 0:
            print(
                f"Encoder: Unfreezing the last {self.unfreeze_layers} layers of Vision Transformer."
            )
            for param in self.swin.parameters():
                param.requires_grad = False
            layers = self.swin.encoder.layers
            total_layers = len(layers)
            start_index = max(0, total_layers - self.unfreeze_layers)
            for i in range(start_index, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            self._fully_frozen = False
        else:
            print("Encoder: Vision Encoder is 100% frozen.")
            for param in self.swin.parameters():
                param.requires_grad = False

    def forward(self, pixel_values):
        if self._fully_frozen:
            with torch.no_grad():
                outputs = self.swin(pixel_values)
        else:
            outputs = self.swin(pixel_values)
        return outputs.last_hidden_state


class CaptionModel(nn.Module):
    def __init__(
        self,
        unfreeze_vision=UNFREEZE_VISION_LAYERS,
        unfreeze_bangla=UNFREEZE_BANGLA_LAYERS,
    ):
        super().__init__()
        self.encoder = Encoder(unfreeze_layers=unfreeze_vision)
        self.vision_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
        )

        bert_path = (
            BANGLA_BERT_PATH
            if os.path.exists(BANGLA_BERT_PATH)
            else "csebuetnlp/banglabert"
        )
        config = BertConfig.from_pretrained(bert_path, trust_remote_code=True)
        config.is_decoder = True
        config.add_cross_attention = True
        self.banglabert = BertLMHeadModel.from_pretrained(
            bert_path, config=config, trust_remote_code=True
        )

        for param in self.banglabert.parameters():
            param.requires_grad = False

        if unfreeze_bangla >= 0:
            layers = self.banglabert.bert.encoder.layer
            num_to_unfreeze = (
                len(layers)
                if unfreeze_bangla == 0
                else min(unfreeze_bangla, len(layers))
            )
            for i in range(len(layers) - num_to_unfreeze, len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
            self.banglabert.cls.predictions.decoder.requires_grad_(True)

        self.banglabert.gradient_checkpointing_enable()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, label_smoothing=LABEL_SMOOTHING
        )

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)

        outputs = self.banglabert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_tokens,
            return_dict=True,
        )
        logits = outputs.logits

        if labels is None:
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[attention_mask == 0] = -100

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fn(
            shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1)
        )

        return ModelOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_captions(self, pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        batch_size = pixel_values.size(0)

        self.banglabert.gradient_checkpointing_disable()

        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)

        bos_id = tokenizer.cls_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id
        suppress_ids = {
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.cls_token_id,
        }

        generated = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=pixel_values.device
        )

        for _ in range(max_new_tokens):
            outputs = self.banglabert(
                input_ids=generated,
                encoder_hidden_states=visual_tokens,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]

            for sid in suppress_ids:
                if sid is not None:
                    logits[:, sid] = -float("inf")

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        self.banglabert.gradient_checkpointing_enable()

        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]

    @torch.no_grad()
    def generate_caption(self, pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN):
        return self.generate_captions(pixel_values, tokenizer, max_new_tokens)[0]


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
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))
    train_list = load_image_list(os.path.join(DATA_DIR, "caption", "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "caption", "validation.txt"))

    total = len(train_list) + len(val_list)
    print(f"\n{'=' * 50}")
    print(f"{'Split':<12} {'Count':<10} {'Percentage':<10}")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"{'Train':<12} {len(train_list):<10} {len(train_list) / total * 100:.2f}%")
    print(f"{'Val':<12} {len(val_list):<10} {len(val_list) / total * 100:.2f}%")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"{'Total':<12} {total:<10} {'100.00%'}")
    print(f"{'=' * 50}\n")
    print(f"Captions: {len(captions)}")

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

    model = CaptionModel().to(DEVICE)

    vision_params = [p for p in model.encoder.parameters() if p.requires_grad]
    projection_params = list(model.vision_projection.parameters())
    bangla_params = [p for p in model.banglabert.parameters() if p.requires_grad]
    all_params = vision_params + projection_params + bangla_params

    optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=max(1, len(train_loader) // ACCUMULATION_STEPS),
        epochs=EPOCHS,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    total_trainable = sum(p.numel() for p in all_params)
    unfrozen_vision = sum(p.requires_grad for p in model.encoder.swin.parameters())
    vision_status = (
        "Fully Frozen"
        if UNFREEZE_VISION_LAYERS < 0
        else (
            "All Trainable"
            if UNFREEZE_VISION_LAYERS == 0
            else f"Last {UNFREEZE_VISION_LAYERS} Layers Unfrozen ({unfrozen_vision} params)"
        )
    )
    unfrozen_bangla_count = sum(p.requires_grad for p in model.banglabert.parameters())
    bangla_status = (
        "Fully Frozen"
        if UNFREEZE_BANGLA_LAYERS < 0
        else (
            "All Trainable"
            if UNFREEZE_BANGLA_LAYERS == 0
            else f"Last {UNFREEZE_BANGLA_LAYERS} Layers Unfrozen ({unfrozen_bangla_count} params)"
        )
    )
    print(f"\nArchitecture: Swin-tiny Vision -> BanglaBERT Decoder")
    print(f"Vision: {vision_status} | BanglaBERT: {bangla_status}")
    print(f"Trainable params: {total_trainable:,}")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE * ACCUMULATION_STEPS}\n")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

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
                },
                os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(OUTPUT_DIR, "final_model.pt"),
    )
    print("Training complete!")


if __name__ == "__main__":
    download_models()
    train()
