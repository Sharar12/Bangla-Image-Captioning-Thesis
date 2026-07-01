import os
import sys
import io
import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModel,
    BertLMHeadModel,
    BertConfig,
)
from PIL import Image
from flask import Flask, request, render_template_string

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")
MODELS_DIR = os.path.join(MODEL_FOLDER, "models")
SWIN_MODEL_PATH = os.path.join(MODELS_DIR, "swin-tiny-patch4-window7-224")
BANGLA_BERT_PATH = os.path.join(MODELS_DIR, "banglabert")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN = 51
FEATURE_DIM = 768
HIDDEN_SIZE = 768
DROPOUT = 0.1

app = Flask(__name__)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        swin_path = (
            SWIN_MODEL_PATH
            if os.path.exists(SWIN_MODEL_PATH)
            else "microsoft/swin-tiny-patch4-window7-224"
        )
        self.swin = AutoModel.from_pretrained(swin_path, trust_remote_code=True)
        for param in self.swin.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.swin(pixel_values)
        return outputs.last_hidden_state


class CaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.vision_projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
        )

    @torch.no_grad()
    def generate_caption(self, pixel_values, tokenizer, max_new_tokens=MAX_SEQ_LEN):
        self.eval()
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        features = self.encoder(pixel_values).float()
        visual_tokens = self.vision_projection(features)

        eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id

        outputs = self.banglabert.generate(
            inputs_embeds=visual_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

        sequences = outputs.sequences
        return tokenizer.decode(sequences[0], skip_special_tokens=True)


checkpoint_path = os.path.join(OUTPUT_DIR, "final_model.pt")
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(OUTPUT_DIR, "final_model_4bit.pt")
print(f"Loading model from: {checkpoint_path}")
print(f"Device: {DEVICE}")

current_module = sys.modules[__name__]
sys.modules["__main__"] = current_module

checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

model = CaptionModel()

swin_path = (
    SWIN_MODEL_PATH
    if os.path.exists(SWIN_MODEL_PATH)
    else "microsoft/swin-tiny-patch4-window7-224"
)
swin = AutoModel.from_pretrained(swin_path, trust_remote_code=True)
for param in swin.parameters():
    param.requires_grad = False
setattr(model.encoder, "swin", swin)

config = BertConfig.from_pretrained(
    BANGLA_BERT_PATH if os.path.exists(BANGLA_BERT_PATH) else "csebuetnlp/banglabert",
    trust_remote_code=True,
)
config.is_decoder = True
banglabert = BertLMHeadModel.from_pretrained(
    BANGLA_BERT_PATH if os.path.exists(BANGLA_BERT_PATH) else "csebuetnlp/banglabert",
    config=config,
    trust_remote_code=True,
)
for param in banglabert.parameters():
    param.requires_grad = False
setattr(model, "banglabert", banglabert)

model = model.to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

processor_path = (
    swin_path
    if os.path.exists(os.path.join(swin_path, "preprocessor_config.json"))
    else "microsoft/swin-tiny-patch4-window7-224"
)
processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    BANGLA_BERT_PATH if os.path.exists(BANGLA_BERT_PATH) else "csebuetnlp/banglabert",
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model is ready!")


def generate_caption(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        with torch.amp.autocast("cuda"):
            caption = model.generate_caption(pixel_values, tokenizer, MAX_SEQ_LEN)

    return caption


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangla Image Captioning - Swin + BanglaBERT</title>
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
        .btn-generate:disabled { background: #475569; cursor: not-allowed; }
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
        .image-panel img { width: 100%; height: auto; border-radius: 12px; display: block; }
        .caption-panel { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center; }
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
        .error { color: #f87171; text-align: center; margin-top: 1rem; display: none; }
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
        <p class="subtitle">Swin-tiny + BanglaBERT</p>

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
