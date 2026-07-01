import pandas as pd
import os
import shutil
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
MASTER_CSV = "hybrid_60k_master.csv"
FINAL_DIR = "Final_Hybrid_Dataset_60k/images"

LOCAL_SOURCES = {
    'Flickr30k': "Source_Images/Flickr_30k_Images",
    'BNature':   "Source_Images/BNature_Images",
    'COCO':      "Source_Images/COCO_Images"
}

def build_dataset():
    if not os.path.exists(MASTER_CSV):
        print(f"❌ Error: {MASTER_CSV} not found.")
        return

    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)

    print("="*60)
    print("🚀 HYBRID 60K DATASET BUILDER")
    print("="*60)
    print("1. [WEB MODE] Download COCO from Web")
    print("2. [LOCAL MODE] Merge everything from local folders")
    print("-" * 60)
    
    choice = input("👉 Enter Choice (1 or 2): ").strip()

    print(f"\n📂 Loading {MASTER_CSV}...")
    try:
        df = pd.read_csv(MASTER_CSV)
        unique_files = df[['source_folder', 'original_filename', 'virtual_filename', 'image_url']].drop_duplicates()
        print(f"📊 Total Images to Process: {len(unique_files)}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    def process_file(row):
        source = row['source_folder']
        orig_name = row['original_filename']
        final_name = row['virtual_filename']
        url = row['image_url']
        
        dest_path = os.path.join(FINAL_DIR, final_name)
        
        if os.path.exists(dest_path):
            return "Skipped"

        local_src_dir = LOCAL_SOURCES.get(source)
        if local_src_dir:
            local_path = os.path.join(local_src_dir, orig_name)
            if os.path.exists(local_path):
                shutil.copy2(local_path, dest_path)
                return "Copied"
        
        if choice == '1' and isinstance(url, str) and url.startswith('http'):
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(r.content)
                    return "Downloaded"
            except:
                pass
        
        return "Missing"

    print("\n🚀 Starting Build Process...")
    rows = df.to_dict('records')
    results = {"Copied": 0, "Downloaded": 0, "Missing": 0, "Skipped": 0}
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        for status in tqdm(executor.map(process_file, rows), total=len(rows)):
            results[status] += 1

    print("\n" + "="*60)
    print("🎉 BUILD COMPLETE!")
    print(f"📂 Final Images: {FINAL_DIR}")
    print(f"✅ Copied: {results['Copied']} | 🌐 Downloaded: {results['Downloaded']}")
    print(f"⏭️ Skipped: {results['Skipped']} | ❌ Missing: {results['Missing']}")
    print("="*60)

if __name__ == "__main__":
    build_dataset()
