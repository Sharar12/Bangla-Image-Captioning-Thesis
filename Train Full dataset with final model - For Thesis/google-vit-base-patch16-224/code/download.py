import os
import sys

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

MODEL_NAME = "google/vit-base-patch16-224"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "model")

def download_with_progress():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Model: {MODEL_NAME}")
    print(f"Target: {OUTPUT_DIR}")
    print("=" * 50)
    print("Note: ViT-base-patch16-224 is ~1GB")
    print("-" * 50)
    print()
    
    print("[1/1] Downloading model files with progress...")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False
    )
    
    print("\n" + "=" * 50)
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
        for f in os.listdir(OUTPUT_DIR) 
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    )
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"Download complete!")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"\nSaved files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}: {size:.2f} MB")

if __name__ == "__main__":
    download_with_progress()
