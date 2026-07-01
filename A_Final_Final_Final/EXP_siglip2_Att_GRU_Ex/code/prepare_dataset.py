import os
import csv
import json
import shutil
from collections import defaultdict
from tqdm import tqdm

PROJECT_ROOT = r"D:\Python Projects\A_Final_Final_Final"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CAPTION_FILE = os.path.join(OUTPUT_DIR, "caption.txt")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "filename_mapping.json")

BANGLAVISION_DIR = os.path.join(PROJECT_ROOT, "BanglaVision40k")
COCO_DIR = os.path.join(PROJECT_ROOT, "coco2017")

SUBDIRS = ["BanglaView", "BNature", "Bornon"]

BANGLAVISION_CSV = os.path.join(BANGLAVISION_DIR, "captions.csv")

V5_DIR = os.path.join(BANGLAVISION_DIR, "V5 Beyond Caption Dataset")
V5_ANNOTATIONS = os.path.join(V5_DIR, "annotation.csv")
V5_IMAGES_DIR = os.path.join(V5_DIR, "Images1")
COCO_TRAIN_JSON = os.path.join(
    COCO_DIR, "annotations_Bangla", "captions_train2017_bangla.json"
)
COCO_VAL_JSON = os.path.join(
    COCO_DIR, "annotations_Bangla", "captions_val2017_bangla.json"
)

COCO_TRAIN_IMAGES = os.path.join(COCO_DIR, "train2017")
COCO_VAL_IMAGES = os.path.join(COCO_DIR, "val2017")
COCO_TEST_IMAGES = os.path.join(COCO_DIR, "test2017")


def load_banglavision_csv(csv_path):
    image_to_captions = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"].strip()
            caption = row["caption"].strip()
            image_to_captions[img_path].append(caption)
    return image_to_captions


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


def load_coco_json(json_path, images_dir):
    image_to_captions = defaultdict(list)
    image_id_to_file = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for img in data["images"]:
        image_id_to_file[img["id"]] = img["file_name"]
    for ann in data["annotations"]:
        fname = image_id_to_file[ann["image_id"]]
        image_to_captions[fname].append(ann["caption"].strip())
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
            f"WARNING: Dataset already exists ({len(existing_mapping)} images, {CAPTION_FILE})!"
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

    print("Loading COCO train2017 captions...")
    coco_train_captions = load_coco_json(COCO_TRAIN_JSON, COCO_TRAIN_IMAGES)
    print(f"  {len(coco_train_captions)} unique images loaded")

    print("Loading COCO val2017 captions...")
    coco_val_captions = load_coco_json(COCO_VAL_JSON, COCO_VAL_IMAGES)
    print(f"  {len(coco_val_captions)} unique images loaded")

    new_to_old = dict(existing_mapping)
    counter = iter(range(start_id, 10000000))

    def copy_if_new(src, mapping, counter):
        if os.path.abspath(src) in copied_sources:
            return
        new_id = next(counter)
        new_fname = f"{new_id}.jpg"
        shutil.copy2(src, os.path.join(IMAGES_DIR, new_fname))
        mapping[new_fname] = src

    print("\nCopying BanglaVision40k images...")
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
                continue
            new_id = next(counter)
            new_fname = f"{new_id}.jpg"
            shutil.copy2(src, os.path.join(IMAGES_DIR, new_fname))
            new_to_old[new_fname] = src
        sub_count = len([v for v in new_to_old.values() if src_dir in v])
        print(f"    {sub_count} images in dataset from {subdir}")

    print("Copying COCO train2017 images...")
    for fname in tqdm(coco_train_captions, desc="  train2017"):
        src = os.path.join(COCO_TRAIN_IMAGES, fname)
        if os.path.exists(src):
            copy_if_new(src, new_to_old, counter)
    new_count = len([v for v in new_to_old.values() if COCO_TRAIN_IMAGES in v])
    print(f"    {new_count} images in dataset")

    print("Copying COCO val2017 images...")
    for fname in tqdm(coco_val_captions, desc="  val2017"):
        src = os.path.join(COCO_VAL_IMAGES, fname)
        if os.path.exists(src):
            copy_if_new(src, new_to_old, counter)
    new_count = len([v for v in new_to_old.values() if COCO_VAL_IMAGES in v])
    print(f"    {new_count} images in dataset")

    v5_data = load_v5_annotations() if os.path.exists(V5_ANNOTATIONS) else {}
    print("\nProcessing V5 Beyond Caption Dataset...")
    if v5_data and os.path.isdir(V5_IMAGES_DIR):
        v5_present = [
            f
            for f in os.listdir(V5_IMAGES_DIR)
            if f in v5_data and os.path.isfile(os.path.join(V5_IMAGES_DIR, f))
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

    old_to_new = {}
    for new_fname, src_path in new_to_old.items():
        old_fname = os.path.basename(src_path)
        old_to_new[old_fname] = new_fname

    print("\nWriting caption.txt...")
    caption_count = 0
    with open(CAPTION_FILE, "w", encoding="utf-8") as out_f:
        for new_fname, src_path in tqdm(new_to_old.items(), desc="  writing captions"):
            old_fname = os.path.basename(src_path)
            parent_dir = os.path.basename(os.path.dirname(src_path))
            rel_path = f"{parent_dir}/{old_fname}"

            caps = None
            if parent_dir in SUBDIRS:
                caps = banglavision_captions.get(rel_path, [])
            elif parent_dir == "train2017":
                caps = coco_train_captions.get(old_fname, [])
            elif parent_dir == "val2017":
                caps = coco_val_captions.get(old_fname, [])

            if not caps:
                caps = v5_data.get(old_fname, [])

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
    print(f"BanglaVision40k + V5 Beyond Caption consolidation complete!")
    print(f"  Images: {total_copied}")
    print(f"  Caption lines: {caption_count}")
    print(f"  Images dir: {IMAGES_DIR}")
    print(f"  Caption file: {CAPTION_FILE}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
