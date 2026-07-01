import os
import csv
import json
import shutil
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = r"D:\Python Projects\A_Final_Final_Final"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "dataset40k"
)
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CAPTION_FILE = os.path.join(OUTPUT_DIR, "caption.txt")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "filename_mapping.json")

BANGLAVISION_DIR = os.path.join(PROJECT_ROOT, "BanglaVision40k")
BANGLAVISION_CSV = os.path.join(BANGLAVISION_DIR, "captions.csv")

SUBDIRS = ["BanglaView", "BNature", "Bornon"]


def load_banglavision_csv(csv_path):
    image_to_captions = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"].strip()
            caption = row["caption"].strip()
            image_to_captions[img_path].append(caption)
    return image_to_captions


V5_DIR = os.path.join(BANGLAVISION_DIR, "V5 Beyond Caption Dataset")
V5_ANNOTATIONS = os.path.join(V5_DIR, "annotation.csv")
V5_IMAGES_DIR = os.path.join(V5_DIR, "Images1")


def load_v5_annotations():
    image_to_captions = {}
    with open(V5_ANNOTATIONS, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        temp = defaultdict(list)
        for row in reader:
            img = row["image"].strip()
            cap = row["short_caption_bn"].strip()
            temp[img].append(cap)
    for img, caps in temp.items():
        if len(caps) == 5:
            image_to_captions[img] = caps
    return image_to_captions


def load_existing_mapping():
    if not os.path.exists(MAPPING_FILE):
        return {}, set()
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_to_old = data.get("new_to_old", {})
    copied_sources = set(os.path.abspath(p) for p in new_to_old.values())
    return new_to_old, copied_sources


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    existing_mapping, copied_sources = load_existing_mapping()
    if existing_mapping:
        print(
            f"WARNING: dataset40k already exists ({len(existing_mapping)} images, {CAPTION_FILE})!"
        )
        print("Running again will REGENERATE caption.txt and filename_mapping.json,")
        print("which will UNDO all cleaning (translations, garbage removal, etc.).")
        response = input("Continue? (yes/no): ").strip().lower()
        if response != "yes":
            print("Aborted.")
            return

    start_id = 1
    if existing_mapping:
        existing_ids = [
            int(f.replace(".jpg", "")) for f in existing_mapping if f.endswith(".jpg")
        ]
        start_id = max(existing_ids) + 1 if existing_ids else 1
        print(
            f"Resuming: {len(existing_mapping)} images already copied, starting ID={start_id}"
        )

    print("Loading BanglaVision40k captions...")
    banglavision_captions = load_banglavision_csv(BANGLAVISION_CSV)
    print(f"  {len(banglavision_captions)} unique image keys loaded")

    all_image_keys = set(banglavision_captions.keys())
    print(f"  {len(all_image_keys)} unique images in CSV")

    new_to_old = dict(existing_mapping)
    counter = iter(range(start_id, 10000000))
    skipped = 0

    print("\nCopying images...")
    for subdir in SUBDIRS:
        src_dir = os.path.join(BANGLAVISION_DIR, subdir)
        if not os.path.isdir(src_dir):
            print(f"  Warning: {src_dir} not found, skipping")
            continue
        present = [
            f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))
        ]
        for fname in tqdm(present, desc=f"  {subdir}"):
            rel_path = f"{subdir}/{fname}"
            if rel_path not in all_image_keys:
                continue
            src = os.path.join(src_dir, fname)
            if os.path.abspath(src) in copied_sources:
                continue
            if not os.path.exists(src):
                skipped += 1
                continue
            new_id = next(counter)
            new_fname = f"{new_id}.jpg"
            shutil.copy2(src, os.path.join(IMAGES_DIR, new_fname))
            new_to_old[new_fname] = src
        sub_count = len([v for v in new_to_old.values() if src_dir in v])
        print(f"    {sub_count} images in dataset from {subdir}")

    print("\nProcessing V5 Beyond Caption Dataset...")
    if os.path.isdir(V5_IMAGES_DIR) and os.path.exists(V5_ANNOTATIONS):
        v5_captions = load_v5_annotations()
        print(f"  {len(v5_captions)} images with 5 short captions")
        v5_present = [
            f
            for f in os.listdir(V5_IMAGES_DIR)
            if f in v5_captions and os.path.isfile(os.path.join(V5_IMAGES_DIR, f))
        ]
        for fname in tqdm(v5_present, desc="  V5 Beyond Caption"):
            src = os.path.join(V5_IMAGES_DIR, fname)
            if os.path.abspath(src) in copied_sources:
                continue
            new_id = next(counter)
            new_fname = f"{new_id}.jpg"
            shutil.copy2(src, os.path.join(IMAGES_DIR, new_fname))
            new_to_old[new_fname] = src
        print(
            f"    {len([v for v in new_to_old.values() if V5_IMAGES_DIR in v])} images in dataset from V5 Beyond Caption"
        )
    else:
        print("  V5 Beyond Caption Dataset not found, skipping")

    total_copied = len(new_to_old)
    print(f"\nTotal images in dataset: {total_copied}")
    if skipped:
        print(f"  ({skipped} files skipped due to missing source)")

    old_to_new = {}
    for new_fname, src_path in new_to_old.items():
        old_fname = os.path.basename(src_path)
        old_to_new[old_fname] = new_fname

    print("\nWriting caption.txt...")
    v5_captions = load_v5_annotations() if os.path.exists(V5_ANNOTATIONS) else {}
    caption_count = 0
    with open(CAPTION_FILE, "w", encoding="utf-8") as out_f:
        for new_fname, src_path in tqdm(new_to_old.items(), desc="  writing captions"):
            old_fname = os.path.basename(src_path)
            parent_dir = os.path.basename(os.path.dirname(src_path))
            rel_path = f"{parent_dir}/{old_fname}"
            caps = banglavision_captions.get(rel_path, [])
            if not caps:
                caps = v5_captions.get(old_fname, [])
            if caps:
                for cap in caps:
                    out_f.write(f"{new_fname} {cap}\n")
                    caption_count += 1

    print(f"  {caption_count} total caption lines written")

    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"old_to_new": old_to_new, "new_to_old": new_to_old},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Filename mapping saved to {MAPPING_FILE}")

    print(f"\n{'=' * 50}")
    print(f"dataset40k consolidation complete!")
    print(f"  Images: {total_copied}")
    print(f"  Caption lines: {caption_count}")
    print(f"  Images dir: {IMAGES_DIR}")
    print(f"  Caption file: {CAPTION_FILE}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
