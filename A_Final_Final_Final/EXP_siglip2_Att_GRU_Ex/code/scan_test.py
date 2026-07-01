import os
import sys
import gc
import json
import math
import warnings
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")
BANGLA_GPT_PATH = os.path.join(MODELS_DIR, "BanglaGPT")
os.environ["HF_HOME"] = os.path.join(MODEL_FOLDER, "hf_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 26
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.1
BATCH_SIZE = 8
GRU_NUM_LAYERS = 4
LSTM_NUM_LAYERS = 4
BANGLA_GPT_HIDDEN = 768

TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05
TRAIN_SEED = 8316


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.count = 4

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
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=GRU_NUM_LAYERS, batch_first=True)
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
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=LSTM_NUM_LAYERS, batch_first=True)
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
        model_path = BANGLA_GPT_PATH if os.path.exists(BANGLA_GPT_PATH) else "shahidul034/BanglaGPT"
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
    def __init__(self):
        super().__init__()
        model_path = SIGLIP_MODEL_PATH if os.path.exists(SIGLIP_MODEL_PATH) else "google/siglip2-base-patch32-256"
        self.siglip = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        for param in self.siglip.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        return self.siglip.vision_model(pixel_values).last_hidden_state


class CaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.decoder = MultiDecoderFusion(vocab_size)

    def forward(self, pixel_values, input_ids):
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        return self.decoder(visual_tokens, input_ids)

    @torch.no_grad()
    def generate_captions(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        start_id, end_id = vocab.word2idx["<start>"], vocab.word2idx["<end>"]
        generated = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        unk_id = vocab.word2idx.get("<unk>", 3)
        for _ in range(max_new_tokens - 1):
            logits = self.decoder(visual_tokens, generated)
            logits[:, -1, unk_id] = -1e9
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return [vocab.decode(seq.tolist()) for seq in generated]

    @torch.no_grad()
    def generate_beam_batch(
        self, pixel_values, vocab, beam_size=5, max_new_tokens=MAX_SEQ_LEN
    ):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        batch_size = pixel_values.size(0)
        device = pixel_values.device

        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)

        start_id = vocab.word2idx["<start>"]
        end_id = vocab.word2idx["<end>"]

        visual_tokens = visual_tokens.repeat_interleave(beam_size, dim=0)

        sequences = torch.full(
            (batch_size * beam_size, 1), start_id, dtype=torch.long, device=device
        )
        beam_scores = torch.full((batch_size, beam_size), -1e9, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        done_mask = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens - 1):
            if done_mask.all():
                break

            logits = self.decoder(visual_tokens, sequences)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            log_probs[done_mask] = -1e9
            log_probs[done_mask, vocab.word2idx["<pad>"]] = 0.0

            log_probs[:, vocab.word2idx.get("<unk>", 3)] = -1e9

            next_scores = beam_scores.unsqueeze(-1) + log_probs
            next_scores = next_scores.view(batch_size, -1)

            topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)

            beam_scores = topk_scores.view(-1)

            beam_ids = topk_indices // vocab.vocab_size
            token_ids = topk_indices % vocab.vocab_size

            batch_offsets = torch.arange(
                0, batch_size * beam_size, beam_size, device=device
            ).unsqueeze(-1)
            flat_beam_ids = (beam_ids + batch_offsets).view(-1)

            sequences = torch.cat(
                [sequences[flat_beam_ids], token_ids.view(-1, 1)], dim=-1
            )
            done_mask = done_mask[flat_beam_ids]

            new_ends = (token_ids == end_id).view(-1)
            done_mask = done_mask | new_ends

        sequences = sequences.view(batch_size, beam_size, -1)
        best_sequences = sequences[:, 0, :]

        return [vocab.decode(seq.tolist()) for seq in best_sequences]


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
    def __init__(self, image_list, captions, processor, image_dir):
        self.image_list = image_list
        self.captions = captions
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
            return None
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            all_caps = self.captions.get(img_name, [""])
            return pixel_values, all_caps, img_name
        except Exception:
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    pixel_values = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    img_names = [item[2] for item in batch]
    return pixel_values, captions, img_names


def calculate_cider(reference_list, hypothesis):
    def get_ngrams(tokens, n):
        return list(ngrams(tokens, n))

    def cosine_similarity(ref_ngrams, hyp_ngrams):
        ref_freq, hyp_freq = defaultdict(int), defaultdict(int)
        for ng in ref_ngrams:
            ref_freq[ng] += 1
        for ng in hyp_ngrams:
            hyp_freq[ng] += 1
        all_ngrams = set(ref_freq.keys()) | set(hyp_freq.keys())
        ref_vec = [ref_freq.get(ng, 0) for ng in all_ngrams]
        hyp_vec = [hyp_freq.get(ng, 0) for ng in all_ngrams]
        ref_norm = math.sqrt(sum(x**2 for x in ref_vec))
        hyp_norm = math.sqrt(sum(x**2 for x in hyp_vec))
        if ref_norm == 0 or hyp_norm == 0:
            return 0.0
        return sum(ref_vec[i] * hyp_vec[i] for i in range(len(all_ngrams))) / (ref_norm * hyp_norm)

    hypothesis = hypothesis.replace("।", "")
    hyp_tokens = nltk.word_tokenize(hypothesis)
    scores = []
    for ref_tokens in [nltk.word_tokenize(ref.replace("।", "")) for ref in reference_list]:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens):
                continue
            score += cosine_similarity(get_ngrams(ref_tokens, n), get_ngrams(hyp_tokens, n))
        scores.append(score / 4.0)
    return max(scores) if scores else 0.0


def calculate_metrics(reference_list, hypothesis):
    hypothesis = hypothesis.replace("।", "")
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    list_of_refs_tokens = [nltk.word_tokenize(ref.replace("।", "")) for ref in reference_list]
    smoothing = SmoothingFunction().method4

    bleu1 = sentence_bleu(list_of_refs_tokens, hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(list_of_refs_tokens, hypothesis_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu(list_of_refs_tokens, hypothesis_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(list_of_refs_tokens, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    try:
        meteor = meteor_score(list_of_refs_tokens, hypothesis_tokens)
    except Exception:
        meteor = 0.0

    def rouge_l(ref_tokens, hyp_tokens):
        ref_set, hyp_set = set(ref_tokens), set(hyp_tokens)
        if not ref_set or not hyp_set:
            return 0.0
        overlap = len(ref_set & hyp_set)
        precision = overlap / len(hyp_set)
        recall = overlap / len(ref_set)
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    rouge_scores = [rouge_l(nltk.word_tokenize(ref.replace("।", "")), hypothesis_tokens) for ref in reference_list]
    cider = calculate_cider(reference_list, hypothesis)

    return {
        "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
        "meteor": meteor, "rouge_l": max(rouge_scores) if rouge_scores else 0.0,
        "cider": cider,
    }


def main():
    print(f"Device: {DEVICE}")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found at {vocab_path}")
        return
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary: {vocab.count} tokens")

    best_model_path = os.path.join(OUTPUT_DIR, "best_model_old.pt")
    if not os.path.exists(best_model_path):
        print(f"Model not found at {best_model_path}, trying checkpoints...")
        import re
        pattern = re.compile(r"checkpoint_epoch_(\d+)\.pt")
        checkpoints = [(int(m.group(1)), os.path.join(OUTPUT_DIR, m.group(0)))
                       for f in os.listdir(OUTPUT_DIR) if (m := pattern.match(f))]
        if not checkpoints:
            print("No model found!")
            return
        checkpoints.sort(key=lambda x: x[0])
        _, best_model_path = checkpoints[-1]
        print(f"Using checkpoint: {best_model_path}")

    print(f"Loading: {best_model_path}")
    current_module = sys.modules[__name__]
    sys.modules["__main__"] = current_module
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)

    model = CaptionModel(vocab_size=vocab.vocab_size).to(DEVICE)
    raw_sd = checkpoint["model_state_dict"]
    clean_sd = {}
    for k, v in raw_sd.items():
        clean_sd[k[10:] if k.startswith("_orig_mod.") else k] = v
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.eval()

    processor_path = (SIGLIP_MODEL_PATH if os.path.exists(os.path.join(SIGLIP_MODEL_PATH, "preprocessor_config.json"))
                      else "google/siglip2-base-patch32-256")
    processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

    captions = load_captions(os.path.join(DATA_DIR, "caption.txt"))
    image_names = list(captions.keys())
    n = len(image_names)

    generator = torch.Generator().manual_seed(TRAIN_SEED)
    perm = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_names = [image_names[i] for i in perm[:n_train]]
    val_names = [image_names[i] for i in perm[n_train:n_train + n_val]]
    test_names = [image_names[i] for i in perm[n_train + n_val:]]

    print(f"\nTest images: {len(test_names)}")

    test_dataset = BanglaCaptionDataset(test_names, captions, processor, os.path.join(DATA_DIR, "images"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_images_perfect = []
    all_results = []

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Scanning test"):
            if not batch:
                continue
            pixel_values, batch_caps, img_names = batch
            pixel_values = pixel_values.to(DEVICE)
            gen_caps = model.generate_beam_batch(pixel_values, vocab, beam_size=5)

            for idx, img_name in enumerate(img_names):
                ref_list = batch_caps[idx]
                gen_cap = gen_caps[idx].strip()
                m = calculate_metrics(ref_list, gen_cap)
                perfect = m["bleu4"] >= 0.9999

                entry = {
                    "img_name": img_name,
                    "gen_caption": gen_cap,
                    "references": ref_list,
                    "bleu1": m["bleu1"], "bleu2": m["bleu2"], "bleu3": m["bleu3"],
                    "bleu4": m["bleu4"], "meteor": m["meteor"], "rouge_l": m["rouge_l"],
                    "cider": m["cider"],
                    "all_1.0": perfect,
                }
                all_results.append(entry)
                if perfect:
                    all_images_perfect.append(img_name)

    print(f"\n{'=' * 120}")
    print(f"RESULTS: {len(all_images_perfect)} / {len(test_names)} images have BLEU-4 = 1.0")
    print(f"{'=' * 120}")

    if all_images_perfect:
        print(f"\nPerfect images ({len(all_images_perfect)}):")
        for name in all_images_perfect:
            print(f"  {name}")
    else:
        print("\nNo images with BLEU-4 = 1.0")

    print(f"\n{'=' * 120}")
    header = f"{'Image':<14} {'BLEU-1':<8} {'BLEU-2':<8} {'BLEU-3':<8} {'BLEU-4':<8} {'METEOR':<8} {'ROUGE-L':<8} {'CIDEr':<8}"
    print(header)
    print("-" * 120)
    for r in all_results:
        marker = " ✓" if r["all_1.0"] else ""
        print(f"{r['img_name']:<14} {r['bleu1']:<8.4f} {r['bleu2']:<8.4f} {r['bleu3']:<8.4f} {r['bleu4']:<8.4f} {r['meteor']:<8.4f} {r['rouge_l']:<8.4f} {r['cider']:<8.4f}{marker}")

    perfect_results = [r for r in all_results if r["all_1.0"]]
    result_file = os.path.join(OUTPUT_DIR, "scan_test_results.json")
    tmp_out = result_file + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(perfect_results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_out, result_file)
    print(f"\nSaved {len(perfect_results)} images with BLEU-4=1.0 to {result_file}")


if __name__ == "__main__":
    main()
