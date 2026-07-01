import os
import time
import random
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\Python Projects\Research Papers Test\BNNATURE"
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_MODEL_PATH = os.path.join(MODELS_DIR, "BanglaGPT")
HF_TOKEN = "YOUR_HF_TOKEN"

BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
EPOCHS = 20
MAX_LR = 1e-4
MAX_SEQ_LEN = 48

USE_AMP = True
NUM_WORKERS = 8

FEATURE_DIM = 768
DECODER_HIDDEN_SIZE = 768
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
MIN_FREQ = 5
UNFREEZE_VISION_LAYERS = 1
UNFREEZE_DECODER_LAYERS = 1


def download_bangla_gpt():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(BANGLA_GPT_MODEL_PATH):
        print(f"BanglaGPT already exists at {BANGLA_GPT_MODEL_PATH}")
        return BANGLA_GPT_MODEL_PATH

    print("Downloading BanglaGPT...")
    tokenizer = AutoTokenizer.from_pretrained(
        "shahidul034/BanglaGPT", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "shahidul034/BanglaGPT",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    tokenizer.save_pretrained(BANGLA_GPT_MODEL_PATH)
    model.save_pretrained(BANGLA_GPT_MODEL_PATH)
    print(f"BanglaGPT saved to {BANGLA_GPT_MODEL_PATH}")
    return BANGLA_GPT_MODEL_PATH


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
            if line:
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
                [transforms.Resize((256, 256)), transforms.ToTensor(), normalize]
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
        model_path = (
            SIGLIP_MODEL_PATH
            if os.path.exists(SIGLIP_MODEL_PATH)
            else "google/siglip2-base-patch32-256"
        )
        self.siglip = AutoModel.from_pretrained(
            model_path, token=HF_TOKEN, trust_remote_code=True
        )

        for param in self.siglip.parameters():
            param.requires_grad = False

        if unfreeze_layers >= 0:
            layers = self.siglip.vision_model.encoder.layers
            num_to_unfreeze = (
                len(layers)
                if unfreeze_layers == 0
                else min(unfreeze_layers, len(layers))
            )
            for i in range(len(layers) - num_to_unfreeze, len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True

        self._fully_frozen = unfreeze_layers < 0

    def forward(self, pixel_values):
        if self._fully_frozen:
            with torch.no_grad():
                outputs = self.siglip.vision_model(pixel_values)
        else:
            outputs = self.siglip.vision_model(pixel_values)
        return outputs.last_hidden_state


class CaptionModel(nn.Module):
    def __init__(
        self,
        unfreeze_vision=UNFREEZE_VISION_LAYERS,
        unfreeze_decoder=UNFREEZE_DECODER_LAYERS,
    ):
        super().__init__()
        self.encoder = Encoder(unfreeze_layers=unfreeze_vision)
        self.vision_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, DECODER_HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
        )

        self.decoder = AutoModelForCausalLM.from_pretrained(
            BANGLA_GPT_MODEL_PATH,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        for param in self.decoder.parameters():
            param.requires_grad = False

        if unfreeze_decoder >= 0:
            layers = self.decoder.transformer.h
            num_to_unfreeze = (
                len(layers)
                if unfreeze_decoder == 0
                else min(unfreeze_decoder, len(layers))
            )
            for i in range(len(layers) - num_to_unfreeze, len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
            self.decoder.lm_head.requires_grad_(True)
            if unfreeze_decoder == 0:
                self.decoder.transformer.wte.requires_grad_(True)

        self.decoder.gradient_checkpointing_enable()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100, label_smoothing=LABEL_SMOOTHING
        )

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        text_embeds = self.decoder.get_input_embeddings()(input_ids)

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
            labels = input_ids
        full_labels = torch.full(
            (batch_size, num_vis + seq_len),
            -100,
            device=input_ids.device,
            dtype=torch.long,
        )
        full_labels[:, num_vis:] = labels

        outputs = self.decoder(
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

    @torch.no_grad()
    def generate_captions(self, pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        batch_size = pixel_values.size(0)

        self.decoder.gradient_checkpointing_disable()

        features = self.encoder(pixel_values)
        visual_tokens = self.vision_projection(features)
        num_vis = visual_tokens.size(1)

        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id

        generated = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=pixel_values.device
        )

        for _ in range(max_new_tokens):
            text_embeds = self.decoder.get_input_embeddings()(generated)
            combined = torch.cat([visual_tokens, text_embeds], dim=1)
            attn_mask = torch.ones(
                batch_size, combined.size(1), device=pixel_values.device
            )
            outputs = self.decoder(
                inputs_embeds=combined, attention_mask=attn_mask, return_dict=True
            )
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        self.decoder.gradient_checkpointing_enable()

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
    log_file = os.path.join(
        OUTPUT_DIR, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

    print("Loading data...")
    train_list = load_image_list(os.path.join(DATA_DIR, "caption", "train.txt"))
    val_list = load_image_list(os.path.join(DATA_DIR, "caption", "validation.txt"))
    captions = load_captions(os.path.join(DATA_DIR, "caption", "caption.txt"))

    total = len(train_list) + len(val_list)
    train_pct = 100 * len(train_list) / total if total > 0 else 0
    val_pct = 100 * len(val_list) / total if total > 0 else 0
    print(f"{'Split':<12} {'Count':<8} {'Percentage':<12}")
    print("-" * 32)
    print(f"{'Train':<12} {len(train_list):<8} {train_pct:.2f}%")
    print(f"{'Validation':<12} {len(val_list):<8} {val_pct:.2f}%")
    print(f"{'Total':<12} {total:<8} {100:.2f}%")
    print(f"Captions: {len(captions)}\n")

    print("Loading BanglaGPT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BANGLA_GPT_MODEL_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
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
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    all_params = vision_params + projection_params + decoder_params

    optimizer = torch.optim.AdamW(all_params, lr=MAX_LR, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
        epochs=EPOCHS,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    total_trainable = sum(p.numel() for p in all_params)
    unfrozen_vision = sum(
        p.requires_grad for p in model.encoder.siglip.vision_model.parameters()
    )
    vision_status = (
        "Fully Frozen"
        if UNFREEZE_VISION_LAYERS < 0
        else (
            "All Trainable"
            if UNFREEZE_VISION_LAYERS == 0
            else f"Last {UNFREEZE_VISION_LAYERS} Layers Unfrozen ({unfrozen_vision} params)"
        )
    )
    unfrozen_decoder_count = sum(p.requires_grad for p in model.decoder.parameters())
    decoder_status = (
        "Fully Frozen"
        if UNFREEZE_DECODER_LAYERS < 0
        else (
            "All Trainable"
            if UNFREEZE_DECODER_LAYERS == 0
            else f"Last {UNFREEZE_DECODER_LAYERS} Layers Unfrozen ({unfrozen_decoder_count} params)"
        )
    )
    print(f"\nArchitecture: SigLIP2 Vision -> BanglaGPT Decoder")
    print(f"Vision: {vision_status} | BanglaGPT: {decoder_status}")
    print(f"Trainable params: {total_trainable:,}")
    print(f"Device: {DEVICE}, Batch: {BATCH_SIZE * ACCUMULATION_STEPS}\n")

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
            f"Epoch {epoch + 1}/{EPOCHS} | Time: {epoch_time:.1f}s | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}\n"
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
    download_bangla_gpt()
    train()
