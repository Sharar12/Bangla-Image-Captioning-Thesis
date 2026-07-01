import os
import json
import math
import warnings
from collections import defaultdict
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from flask import Flask, request, jsonify, render_template_string
import base64
from io import BytesIO
import gc
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

app = Flask(__name__)


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Model Architecture (same as upload.py)
# ──────────────────────────────────────────────
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
    def generate_caption(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)
        device = pixel_values.device
        start_id, end_id = vocab.word2idx["<start>"], vocab.word2idx["<end>"]
        generated = torch.full((1, 1), start_id, dtype=torch.long, device=device)

        unk_id = vocab.word2idx.get("<unk>", 3)
        for _ in range(max_new_tokens - 1):
            logits = self.decoder(visual_tokens, generated)
            logits[:, -1, unk_id] = -1e9
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == end_id:
                break
        return vocab.decode(generated[0].tolist())


model = None
processor = None
vocab = None
ref_captions = None


# ──────────────────────────────────────────────
# Load reference captions for scoring
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Metric calculation functions (from claude_test.py)
# ──────────────────────────────────────────────
def calculate_rouge_l(reference, hypothesis):
    ref_set, hyp_set = set(reference), set(hypothesis)
    if not ref_set or not hyp_set:
        return 0.0
    overlap = len(ref_set & hyp_set)
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )


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
        return sum(ref_vec[i] * hyp_vec[i] for i in range(len(all_ngrams))) / (
            ref_norm * hyp_norm
        )

    hypothesis = hypothesis.replace("\u0964", "")
    hyp_tokens = nltk.word_tokenize(hypothesis)
    scores = []
    for ref_tokens in [
        nltk.word_tokenize(ref.replace("\u0964", "")) for ref in reference_list
    ]:
        score = 0.0
        for n in [1, 2, 3, 4]:
            if n > len(hyp_tokens) or n > len(ref_tokens):
                continue
            score += cosine_similarity(
                get_ngrams(ref_tokens, n), get_ngrams(hyp_tokens, n)
            )
        scores.append(score / 4.0)
    return max(scores) if scores else 0.0


def calculate_metrics(reference_list, hypothesis):
    hypothesis = hypothesis.replace("\u0964", "")
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    list_of_refs_tokens = [
        nltk.word_tokenize(ref.replace("\u0964", "")) for ref in reference_list
    ]
    smoothing = SmoothingFunction().method4

    bleu1 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing,
    )
    bleu2 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing,
    )
    bleu3 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smoothing,
    )
    bleu4 = sentence_bleu(
        list_of_refs_tokens,
        hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    try:
        meteor = meteor_score(list_of_refs_tokens, hypothesis_tokens)
    except Exception:
        meteor = 0.0

    rouge_scores = [
        calculate_rouge_l(
            nltk.word_tokenize(ref.replace("\u0964", "")), hypothesis_tokens
        )
        for ref in reference_list
    ]

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge_l": max(rouge_scores) if rouge_scores else 0.0,
        "cider": calculate_cider(reference_list, hypothesis),
    }


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
def load_model():
    global model, processor, vocab, ref_captions

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.vocab_size} tokens")

    # Load reference captions for scoring
    caption_file = os.path.join(DATA_DIR, "captions.txt")
    if os.path.exists(caption_file):
        ref_captions = load_captions(caption_file)
        print(f"Loaded reference captions: {len(ref_captions)} images")
    else:
        print(f"Warning: No reference captions found at {caption_file}")
        ref_captions = {}

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

    processor_path = (
        siglip_path
        if os.path.exists(os.path.join(siglip_path, "preprocessor_config.json"))
        else "google/siglip2-base-patch32-256"
    )
    processor = AutoImageProcessor.from_pretrained(
        processor_path, trust_remote_code=True
    )

    best_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_10_Final.pt")
    if os.path.exists(best_path):
        print(f"Loading best model: {best_path}")
        checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
        raw_sd = checkpoint["model_state_dict"]
        clean_sd = {}
        for k, v in raw_sd.items():
            clean_sd[k[10:] if k.startswith("_orig_mod.") else k] = v
        model.load_state_dict(clean_sd, strict=False)
    else:
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
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            raw_sd = checkpoint["model_state_dict"]
            clean_sd = {}
            for k, v in raw_sd.items():
                clean_sd[k[10:] if k.startswith("_orig_mod.") else k] = v
            model.load_state_dict(clean_sd, strict=False)
        else:
            raise FileNotFoundError(f"No model checkpoint found in {OUTPUT_DIR}")

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")


# ──────────────────────────────────────────────
# HTML Template
# ──────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla Caption Analyzer - SigLIP2+GRU+LSTM+BanglaGPT</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            display: flex; justify-content: center; align-items: flex-start; padding: 30px 20px;
        }
        .container {
            background: #1e293b; border-radius: 24px; padding: 40px;
            max-width: 900px; width: 100%;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            border: 1px solid #334155;
        }
        h1 { text-align: center; color: #f1f5f9; font-size: 26px; margin-bottom: 4px; font-weight: 600; }
        .subtitle { text-align: center; color: #94a3b8; font-size: 13px; margin-bottom: 24px; }

        .name-input-row {
            display: flex; gap: 12px; margin-bottom: 16px; align-items: center;
        }
        .name-input-row label {
            color: #cbd5e1; font-size: 14px; font-weight: 500; white-space: nowrap;
        }
        .name-input-row input {
            flex: 1; padding: 10px 14px; border-radius: 10px; border: 1px solid #475569;
            background: #0f172a; color: #e2e8f0; font-size: 14px; outline: none;
            transition: border-color 0.2s;
        }
        .name-input-row input:focus { border-color: #3b82f6; }
        .name-input-row input::placeholder { color: #64748b; }

        .upload-area {
            border: 2px dashed #475569; border-radius: 16px; padding: 32px;
            text-align: center; cursor: pointer; transition: all 0.3s ease;
            background: #0f172a;
        }
        .upload-area:hover { border-color: #3b82f6; background: #1a2332; }
        .upload-area.dragover { border-color: #3b82f6; background: #1e3a5f; }
        .upload-icon { font-size: 40px; margin-bottom: 12px; }
        .upload-text { color: #94a3b8; font-size: 14px; }
        .upload-text span { color: #3b82f6; font-weight: 600; }

        .preview-area { display: none; margin-top: 20px; text-align: center; }
        .preview-area img { max-width: 100%; max-height: 360px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }

        .loading { display: none; margin-top: 20px; text-align: center; color: #94a3b8; }
        .spinner { width: 36px; height: 36px; border: 4px solid #334155; border-top: 4px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .result-area { display: none; margin-top: 20px; }

        .caption-card {
            padding: 16px 20px; background: #0f172a; border-radius: 12px;
            border-left: 4px solid #3b82f6; margin-bottom: 16px;
        }
        .caption-card h3 { color: #3b82f6; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .caption-card p { color: #e2e8f0; font-size: 17px; line-height: 1.6; font-weight: 500; }

        .scores-card {
            padding: 16px 20px; background: #0f172a; border-radius: 12px;
            border-left: 4px solid #10b981;
        }
        .scores-card h3 { color: #10b981; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }

        .scores-table { width: 100%; border-collapse: collapse; }
        .scores-table th, .scores-table td { padding: 8px 12px; text-align: center; font-size: 14px; }
        .scores-table th { color: #94a3b8; font-weight: 500; border-bottom: 1px solid #334155; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        .scores-table td { color: #e2e8f0; font-weight: 600; font-size: 16px; }
        .scores-table tr:not(:last-child) td { border-bottom: 1px solid #1e293b; }

        .references-card {
            margin-top: 12px; padding: 16px 20px; background: #0f172a; border-radius: 12px;
            border-left: 4px solid #f59e0b;
        }
        .references-card h3 { color: #f59e0b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .references-card ol { margin: 0; padding-left: 20px; }
        .references-card li { color: #94a3b8; font-size: 13px; line-height: 1.7; margin-bottom: 2px; }

        .error { display: none; margin-top: 20px; padding: 14px; background: #7f1d1d; border-radius: 12px; color: #fca5a5; text-align: center; }

        .info-badge {
            display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 11px; font-weight: 600; margin-left: 8px;
        }
        .badge-dataset { background: #065f46; color: #6ee7b7; }
        .badge-custom { background: #1e3a5f; color: #93c5fd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bangla Caption Analyzer</h1>
        <p class="subtitle">Upload an image &mdash; if it's from dataset40k, scores are computed automatically</p>

        <div class="name-input-row">
            <label for="imgName">Image name</label>
            <input type="text" id="imgName" placeholder="e.g. 28745.jpg (auto-filled from file)">
        </div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">🖼️</div>
            <p class="upload-text">Drag & drop an image here, or <span>click to browse</span></p>
            <input type="file" id="fileInput" accept="image/*" hidden>
        </div>

        <div class="preview-area" id="previewArea">
            <img id="previewImg" src="" alt="Preview">
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Generating caption &amp; computing scores...</p>
        </div>

        <div class="result-area" id="resultArea">
            <div class="caption-card">
                <h3>Generated Caption (Bangla)</h3>
                <p id="captionText"></p>
            </div>
            <div class="references-card" id="referencesCard" style="display:none;">
                <h3>Reference Captions</h3>
                <ol id="referencesList"></ol>
            </div>
            <div class="scores-card" id="scoresCard" style="display:none;">
                <h3>Evaluation Scores</h3>
                <table class="scores-table">
                    <thead>
                        <tr>
                            <th>BLEU-1</th>
                            <th>BLEU-2</th>
                            <th>BLEU-3</th>
                            <th>BLEU-4</th>
                            <th>METEOR</th>
                            <th>ROUGE-L</th>
                            <th>CIDEr</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td id="scoreB1">-</td>
                            <td id="scoreB2">-</td>
                            <td id="scoreB3">-</td>
                            <td id="scoreB4">-</td>
                            <td id="scoreMeteor">-</td>
                            <td id="scoreRouge">-</td>
                            <td id="scoreCider">-</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="error" id="error">
            <p id="errorText"></p>
        </div>
    </div>
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const imgNameInput = document.getElementById('imgName');
        const previewArea = document.getElementById('previewArea');
        const previewImg = document.getElementById('previewImg');
        const loading = document.getElementById('loading');
        const resultArea = document.getElementById('resultArea');
        const captionText = document.getElementById('captionText');
        const scoresCard = document.getElementById('scoresCard');
        const referencesCard = document.getElementById('referencesCard');
        const referencesList = document.getElementById('referencesList');
        const error = document.getElementById('error');
        const errorText = document.getElementById('errorText');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault(); uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) { showError('Please select an image file.'); return; }
            // Auto-fill image name from filename
            if (!imgNameInput.value) {
                imgNameInput.value = file.name;
            }
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewArea.style.display = 'block';
                resultArea.style.display = 'none';
                error.style.display = 'none';
                analyzeImage(e.target.result, imgNameInput.value || file.name);
            };
            reader.readAsDataURL(file);
        }

        function analyzeImage(base64Data, imageName) {
            loading.style.display = 'block';
            resultArea.style.display = 'none';
            error.style.display = 'none';

            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data, image_name: imageName })
            })
            .then(res => res.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.error) {
                    showError(data.error);
                    return;
                }
                captionText.textContent = data.caption;
                resultArea.style.display = 'block';

                // Show references if available
                if (data.references && data.references.length > 0) {
                    referencesList.innerHTML = data.references.map(r => '<li>' + r + '</li>').join('');
                    referencesCard.style.display = 'block';
                } else {
                    referencesCard.style.display = 'none';
                }

                // Show scores if available
                if (data.scores) {
                    document.getElementById('scoreB1').textContent = data.scores.bleu1.toFixed(4);
                    document.getElementById('scoreB2').textContent = data.scores.bleu2.toFixed(4);
                    document.getElementById('scoreB3').textContent = data.scores.bleu3.toFixed(4);
                    document.getElementById('scoreB4').textContent = data.scores.bleu4.toFixed(4);
                    document.getElementById('scoreMeteor').textContent = data.scores.meteor.toFixed(4);
                    document.getElementById('scoreRouge').textContent = data.scores.rouge_l.toFixed(4);
                    document.getElementById('scoreCider').textContent = data.scores.cider.toFixed(4);
                    scoresCard.style.display = 'block';
                } else {
                    scoresCard.style.display = 'none';
                }
            })
            .catch(() => {
                loading.style.display = 'none';
                showError('Failed to connect to server.');
            });
        }

        function showError(msg) { errorText.textContent = msg; error.style.display = 'block'; }
    </script>
</body>
</html>
"""


# ──────────────────────────────────────────────
# Flask Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

        image_name = data.get("image_name", "").strip()
        # Extract just the filename from path if needed
        image_name = os.path.basename(image_name)

        # Generate caption
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        caption = model.generate_caption(pixel_values, vocab)

        result = {"caption": caption, "image_name": image_name}

        # If image name is provided and found in reference captions, compute scores
        if image_name and image_name in ref_captions:
            refs = ref_captions[image_name]
            if refs:
                scores = calculate_metrics(refs, caption)
                result["scores"] = scores
                result["references"] = refs

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    print("Starting server at http://localhost:8183")
    app.run(host="0.0.0.0", port=8183, debug=False)
