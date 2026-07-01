import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from PIL import Image
from flask import Flask, request, render_template_string
import io
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(BASE_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(MODEL_FOLDER, "model")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output_new")
MAX_SEQ_LEN = 32

app = Flask(__name__)


class Vocabulary:
    def __init__(self, word2idx=None):
        if word2idx:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}
            self.n_words = len(word2idx)
        else:
            self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
            self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            self.n_words = 4

    def decode(self, tokens):
        words = []
        for t in tokens:
            w = self.idx2word.get(t, "<UNK>")
            if w in ["<EOS>", "<PAD>"]:
                break
            if w not in ["<SOS>", "<UNK>"]:
                words.append(w)
        return " ".join(words)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
        for param in self.siglip.parameters():
            param.requires_grad = False
        self.feature_dim = 768

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.siglip.vision_model(pixel_values)
        return outputs.last_hidden_state


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        hidden_dim=2048,
        vocab_size=10000,
        feature_dim=768,
        num_layers=6,
        nhead=12,
        dropout=0.1,
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
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size=vocab_size)

    def forward(self, pixel_values, captions):
        features = self.encoder(pixel_values)
        return self.decoder(features, captions)


# --- LOAD EVERYTHING DIRECTLY OUT OF THE CHECKPOINT ---
checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_20.pt")
print(f"Loading fully standalone model from: {checkpoint_path}")
print(f"Running execution device: {DEVICE}")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

word2idx = checkpoint["vocab"]
vocab = Vocabulary(word2idx)
print(f"Recognized vocabulary layout size: {vocab.n_words}")

# Extracts complete object setup (SigLIP2 Backbone + Projection + Decoder Layers)
model = checkpoint["full_model"].to(DEVICE)
model.eval()

# Loads formatting profile instructions mapping rule profiles
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
print("Complete unified standalone model is operational!")


def generate_caption(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        with torch.amp.autocast("cuda"):
            # Internal structural parts execute directly out of checkpoint properties
            features = model.encoder(pixel_values)
            predictions = model.decoder.greedy_decode(features, MAX_SEQ_LEN)

    generated_caption = vocab.decode(predictions[0].cpu().numpy().tolist())
    return generated_caption


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla Image Captioning - Standalone SigLIP2</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }
        .container { max-width: 1000px; width: 100%; }
        h1 {
            text-align: center;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
            background: linear-gradient(135deg, #34d399, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .upload-box {
            background: #1e293b;
            border: 2px dashed #334155;
            border-radius: 16px;
            padding: 2.5rem 2rem;
            text-align: center;
            transition: border-color 0.3s;
            margin-bottom: 2rem;
        }
        .upload-box:hover { border-color: #34d399; }
        .upload-box.dragover { border-color: #34d399; background: #1e3a2f; }
        .upload-label {
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
            margin-bottom: 1rem;
        }
        .upload-label:hover { background: #059669; }
        #file-input { display: none; }
        #filename {
            display: block;
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 1rem;
            min-height: 1.2rem;
        }
        .btn-generate {
            background: #14b8a6;
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-generate:hover { background: #0d9488; }
        .btn-generate:disabled {
            background: #475569;
            cursor: not-allowed;
        }
        .result-area {
            display: none;
            gap: 2rem;
            background: #1e293b;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #334155;
        }
        .result-area.show { display: flex; }
        .image-panel { flex: 1; min-width: 0; }
        .image-panel img {
            width: 100%;
            height: auto;
            border-radius: 12px;
            display: block;
        }
        .caption-panel {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .caption-panel h2 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 0.75rem;
        }
        .caption-text {
            font-size: 1.3rem;
            line-height: 1.7;
            color: #f1f5f9;
            padding: 1.25rem;
            background: #0f172a;
            border-radius: 12px;
            border-left: 4px solid #14b8a6;
            font-family: 'Noto Sans Bengali', 'Segoe UI', sans-serif;
        }
        .spinner {
            display: none;
            width: 24px; height: 24px;
            border: 3px solid #334155;
            border-top-color: #14b8a6;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem;
        }
        .spinner.show { display: block; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error {
            color: #f87171;
            text-align: center;
            margin-top: 1rem;
            display: none;
        }
        .error.show { block; }
        @media (max-width: 640px) {
            .result-area.show { flex-direction: column; }
            .caption-text { font-size: 1.1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bangla Image Captioning</h1>
        <p class="subtitle">Fully Standalone Checkpoint Pipeline</p>

        <div class="upload-box" id="dropzone">
            <label class="upload-label" for="file-input">Choose Image</label>
            <input type="file" id="file-input" accept="image/*">
            <span id="filename">No file selected</span>
            <div class="spinner" id="spinner"></div>
            <button class="btn-generate" id="generateBtn" disabled>Generate Caption</button>
            <div class="error" id="error"></div>
        </div>

        <div class="result-area" id="resultArea">
            <div class="image-panel">
                <img id="previewImage" src="" alt="Uploaded image">
            </div>
            <div class="caption-panel">
                <h2>Generated Caption</h2>
                <div class="caption-text" id="captionText"></div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const filename = document.getElementById('filename');
        const generateBtn = document.getElementById('generateBtn');
        const resultArea = document.getElementById('resultArea');
        const previewImage = document.getElementById('previewImage');
        const captionText = document.getElementById('captionText');
        const spinner = document.getElementById('spinner');
        const error = document.getElementById('error');
        const dropzone = document.getElementById('dropzone');
        let selectedFile = null;

        fileInput.addEventListener('change', (e) => {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                filename.textContent = selectedFile.name;
                generateBtn.disabled = false;
                error.classList.remove('show');
                const reader = new FileReader();
                reader.onload = (ev) => { previewImage.src = ev.target.result; };
                reader.readAsDataURL(selectedFile);
            } else {
                filename.textContent = 'No file selected';
                generateBtn.disabled = true;
            }
        });

        dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileInput.dispatchEvent(new Event('change'));
            }
        });

        generateBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            generateBtn.disabled = true;
            spinner.classList.add('show');
            error.classList.remove('show');
            resultArea.classList.remove('show');

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const resp = await fetch('/upload', { method: 'POST', body: formData });
                if (!resp.ok) throw new Error('Server error: ' + resp.statusText);
                const data = await resp.json();
                if (data.error) throw new Error(data.error);
                captionText.textContent = data.caption;
                resultArea.classList.add('show');
            } catch (err) {
                error.textContent = 'Error: ' + err.message;
                error.classList.add('show');
            } finally {
                generateBtn.disabled = false;
                spinner.classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return {"error": "No image file provided"}, 400
    file = request.files["image"]
    if file.filename == "":
        return {"error": "No file selected"}, 400
    try:
        image_bytes = file.read()
        caption = generate_caption(image_bytes)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    print(f"Server starting at http://localhost:8182")
    app.run(host="0.0.0.0", port=8182, debug=False, threaded=False)
