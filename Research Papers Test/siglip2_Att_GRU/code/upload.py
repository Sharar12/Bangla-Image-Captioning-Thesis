import os
import json
import sys
import warnings
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from flask import Flask, request, jsonify, render_template_string
import base64
from io import BytesIO
import gc

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SIGLIP_MODEL_PATH = os.path.join(MODELS_DIR, "siglip2-base-patch32-256")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 51
FEATURE_DIM = 768
EMBED_DIM = 512
HIDDEN_DIM = 512
DROPOUT = 0.1

app = Flask(__name__)


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

    def step(self, token_embed, hidden, encoder_outputs):
        decoder_output, hidden = self.gru(token_embed, hidden)
        context = self._attention(decoder_output, encoder_outputs)
        combined = torch.cat([context, decoder_output], dim=-1)
        fused = self.tanh(self.concat(combined))
        logits = self.proj(fused)
        return logits, hidden


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
        self.decoder = BanglaLuongAttentionDecoder(vocab_size=vocab_size)

    @torch.no_grad()
    def generate_caption(self, pixel_values, vocab, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        features = self.encoder(pixel_values).float()
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
        return vocab.decode(generated[0].tolist())


model = None
processor = None
vocab = None


def load_model():
    global model, processor, vocab

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")
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

    processor_path = (
        siglip_path
        if os.path.exists(os.path.join(siglip_path, "preprocessor_config.json"))
        else "google/siglip2-base-patch32-256"
    )
    processor = AutoImageProcessor.from_pretrained(
        processor_path, trust_remote_code=True
    )

    checkpoint_paths = [
        os.path.join(OUTPUT_DIR, "final_model.pt"),
        os.path.join(OUTPUT_DIR, "final_model_4bit.pt"),
    ]
    loaded = False
    for ckpt in checkpoint_paths:
        if os.path.exists(ckpt):
            print(f"Loading checkpoint: {ckpt}")
            checkpoint = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            loaded = True
            break

    if not loaded:
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
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            loaded = True

    if not loaded:
        raise FileNotFoundError(f"No model checkpoint found in {OUTPUT_DIR}")

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla Image Captioning - AttGRU (Luong)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            display: flex; justify-content: center; align-items: center; padding: 20px;
        }
        .container {
            background: #1e293b; border-radius: 24px; padding: 40px;
            max-width: 700px; width: 100%;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            border: 1px solid #334155;
        }
        h1 { text-align: center; color: #f1f5f9; font-size: 28px; margin-bottom: 8px; font-weight: 600; }
        .subtitle { text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 30px; }
        .upload-area {
            border: 2px dashed #475569; border-radius: 16px; padding: 40px;
            text-align: center; cursor: pointer; transition: all 0.3s ease;
            background: #0f172a;
        }
        .upload-area:hover { border-color: #3b82f6; background: #1a2332; }
        .upload-area.dragover { border-color: #3b82f6; background: #1e3a5f; }
        .upload-icon { font-size: 48px; margin-bottom: 16px; }
        .upload-text { color: #94a3b8; font-size: 16px; }
        .upload-text span { color: #3b82f6; font-weight: 600; }
        .preview-area { display: none; margin-top: 24px; text-align: center; }
        .preview-area img { max-width: 100%; max-height: 400px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
        .caption-result { display: none; margin-top: 24px; padding: 20px; background: #0f172a; border-radius: 12px; border-left: 4px solid #3b82f6; }
        .caption-result h3 { color: #3b82f6; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .caption-result p { color: #e2e8f0; font-size: 18px; line-height: 1.6; font-weight: 500; }
        .loading { display: none; margin-top: 24px; text-align: center; color: #94a3b8; }
        .spinner { width: 40px; height: 40px; border: 4px solid #334155; border-top: 4px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 12px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { display: none; margin-top: 24px; padding: 16px; background: #7f1d1d; border-radius: 12px; color: #fca5a5; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bangla Image Captioning</h1>
        <p class="subtitle">SigLIP2 Vision → Luong Attention + GRU Decoder</p>
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
            <p>Generating caption...</p>
        </div>
        <div class="caption-result" id="captionResult">
            <h3>Generated Caption (Bangla)</h3>
            <p id="captionText"></p>
        </div>
        <div class="error" id="error">
            <p id="errorText"></p>
        </div>
    </div>
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewArea = document.getElementById('previewArea');
        const previewImg = document.getElementById('previewImg');
        const loading = document.getElementById('loading');
        const captionResult = document.getElementById('captionResult');
        const captionText = document.getElementById('captionText');
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
            const reader = new FileReader();
            reader.onload = (e) => { previewImg.src = e.target.result; previewArea.style.display = 'block'; captionResult.style.display = 'none'; error.style.display = 'none'; uploadImage(e.target.result); };
            reader.readAsDataURL(file);
        }
        function uploadImage(base64Data) {
            loading.style.display = 'block'; captionResult.style.display = 'none'; error.style.display = 'none';
            fetch('/upload', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: base64Data }) })
            .then(res => res.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.error) showError(data.error);
                else { captionText.textContent = data.caption; captionResult.style.display = 'block'; }
            })
            .catch(() => { loading.style.display = 'none'; showError('Failed to connect to server.'); });
        }
        function showError(msg) { errorText.textContent = msg; error.style.display = 'block'; }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        caption = model.generate_caption(pixel_values, vocab)
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    print("Starting server at http://localhost:8182")
    app.run(host="0.0.0.0", port=8182, debug=False)
