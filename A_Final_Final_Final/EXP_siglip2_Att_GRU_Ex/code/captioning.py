import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_PATH = os.path.join(MODELS_DIR, "BanglaGPT")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 26
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.1
GRU_NUM_LAYERS = 4
LSTM_NUM_LAYERS = 4
BANGLA_GPT_HIDDEN = 768
BANGLA_GPT_NAME = "shahidul034/BanglaGPT"
INFERENCE_BATCH = 64
NUM_WORKERS = 6


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v.word2idx = data["word2idx"]
        v.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return v

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
        return len(self.word2idx)


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
    def forward(self, pixel_values):
        vision_attr = [v for k, v in self._modules.items() if "siglip" in k]
        if vision_attr:
            return vision_attr[0].vision_model(pixel_values).last_hidden_state
        return (
            getattr(self, list(self._modules.keys())[0])
            ._modules["siglip"](pixel_values)
            .last_hidden_state
        )


class CaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = MultiDecoderFusion(vocab_size)

    @torch.no_grad()
    def generate_captions(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        start_id, end_id = vocab.word2idx["<start>"], vocab.word2idx["<end>"]
        generated = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=device
        )

        for _ in range(max_new_tokens - 1):
            logits = self.decoder(visual_tokens, generated)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return [vocab.decode(seq.tolist()) for seq in generated]


class CaptionDataset(Dataset):
    def __init__(self, image_names, image_dir, processor):
        self.image_names = image_names
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixel_values, name


def collate_caption(batch):
    batch = [b for b in batch if b is not None]
    pixel_values = torch.stack([b[0] for b in batch])
    names = [b[1] for b in batch]
    return pixel_values, names


def main():
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.vocab_size} tokens")

    print(f"Loading model on {DEVICE}...")
    model = CaptionModel(vocab_size=vocab.vocab_size)

    siglip_path = (
        SIGLIP_MODEL_PATH
        if os.path.exists(SIGLIP_MODEL_PATH)
        else "google/siglip2-base-patch32-256"
    )
    siglip = AutoModel.from_pretrained(siglip_path, trust_remote_code=True)
    siglip.eval()
    for param in siglip.parameters():
        param.requires_grad = False
    setattr(model.encoder, "siglip", siglip)

    processor = AutoImageProcessor.from_pretrained(siglip_path, trust_remote_code=True)

    ckpt_path = os.path.join(OUTPUT_DIR, "best_model_old.pt")
    if not os.path.exists(ckpt_path):
        import re

        pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt")
        checkpoints = [
            (int(m.group(1)), os.path.join(OUTPUT_DIR, m.group(0)))
            for f in os.listdir(OUTPUT_DIR)
            if (m := pattern.match(f))
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: x[0])
            _, ckpt_path = checkpoints[-1]
        else:
            print(f"No model checkpoint found in {OUTPUT_DIR}")
            return
    print(f"Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    raw_sd = checkpoint["model_state_dict"]
    clean_sd = {}
    for k, v in raw_sd.items():
        clean_sd[k[10:] if k.startswith("_orig_mod.") else k] = v
    model.load_state_dict(clean_sd, strict=False)

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

    image_dir = os.path.join(DATA_DIR, "images")
    image_names = sorted(
        f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))
    )
    print(
        f"Generating captions for {len(image_names)} images (batch={INFERENCE_BATCH})..."
    )

    dataset = CaptionDataset(image_names, image_dir, processor)
    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_caption,
        pin_memory=True,
    )

    output_path = os.path.join(DATA_DIR, "ACCC.txt")
    written = 0
    with open(output_path, "w", encoding="utf-8") as f_out, torch.no_grad():
        for pixel_values, batch_names in tqdm(loader, desc="Generating"):
            pixel_values = pixel_values.to(DEVICE, non_blocking=True)
            captions = model.generate_captions(pixel_values, vocab)
            for name, cap in zip(batch_names, captions):
                cap = cap.strip() or "(খালি)"
                f_out.write(f"{name} {cap}\n")
                written += 1
            f_out.flush()

    print(f"Written: {written} / {len(image_names)} images to {output_path}")


if __name__ == "__main__":
    main()
