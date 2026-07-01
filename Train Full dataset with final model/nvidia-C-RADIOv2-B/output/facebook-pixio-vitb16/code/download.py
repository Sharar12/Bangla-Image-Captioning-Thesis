import os
from huggingface_hub import snapshot_download

MODEL_NAME = "facebook/pixio-vitb16"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
HF_TOKEN = "YOUR_HF_TOKEN"

print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_DIR, token=HF_TOKEN)
print("Download complete!")
