import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template_string
import io
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MAX_SEQ_LEN = 51
FEATURE_DIM = 2048
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.1

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
            if w != "<SOS>":
                words.append(w)
        return " ".join(words)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(weights="DEFAULT")
        self.inception.fc = nn.Identity()
        self.inception.AuxLogits = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False
        self.feature_dim = FEATURE_DIM

    def forward(self, pixel_values):
        with torch.no_grad():
            features = self.inception(pixel_values)
        return features.logits if hasattr(features, "logits") else features


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word2idx = None
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.vision_projection = nn.Linear(feature_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim * 2,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def greedy_decode(self, features, max_len=MAX_SEQ_LEN):
        batch_size = features.size(0)
        proj_feat = self.vision_projection(features).unsqueeze(1)
        hidden = None
        generated = torch.full(
            (batch_size, 1),
            self.word2idx["<SOS>"],
            dtype=torch.long,
        ).to(features.device)
        for _ in range(max_len - 1):
            embed = self.embedding(generated[:, -1:])
            combined = torch.cat([embed, proj_feat], dim=-1)
            combined = self.dropout(combined)
            out, hidden = self.gru(combined, hidden)
            logits = self.fc_out(out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
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


checkpoint_path = os.path.join(OUTPUT_DIR, "final_model.pt")
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(OUTPUT_DIR, "final_model_4bit.pt")
print(f"Loading model from: {checkpoint_path}")
print(f"Device: {DEVICE}")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
vocab_size = checkpoint.get(
    "vocab_size", checkpoint["model_state_dict"]["decoder.embedding.weight"].size(0)
)

model = CaptionModel(vocab_size=vocab_size)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.decoder.word2idx = checkpoint.get("word2idx", None)
if model.decoder.word2idx is None:
    print("Warning: word2idx not found in checkpoint!")
    model.decoder.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

model = model.to(DEVICE)
model.eval()

vocab = Vocabulary(model.decoder.word2idx)
print(f"Model loaded! Vocabulary size: {vocab.n_words}")

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generate_caption(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = transform(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        predictions = model.decoder.greedy_decode(
            model.encoder(pixel_values), MAX_SEQ_LEN
        )

    generated_caption = vocab.decode(predictions[0].cpu().numpy().tolist())
    return generated_caption


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla Image Captioning - InceptionV3 + GRU</title>
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
        .error.show { display: block; }
        @media (max-width: 640px) {
            .result-area.show { flex-direction: column; }
            .caption-text { font-size: 1.1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bangla Image Captioning</h1>
        <p class="subtitle">InceptionV3 + GRU Decoder</p>

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
