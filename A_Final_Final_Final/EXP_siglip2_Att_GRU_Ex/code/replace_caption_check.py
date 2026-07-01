import os
import random
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

RATIO = 0.73

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")

CAPTION_TXT = os.path.join(DATA_DIR, "caption.txt")
ACCC_TXT = os.path.join(DATA_DIR, "ACCC.txt")
ACCC_MODIFIED_TXT = os.path.join(DATA_DIR, "ACCC_modified.txt")
OUTPUT_TXT = os.path.join(DATA_DIR, "captions.txt")


def load_pairs(path):
    pairs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                img, cap = parts[0], parts[1]
                pairs[img] = cap
    return pairs


def load_multi_pairs(path):
    pairs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                img, cap = parts[0], parts[1]
                pairs.setdefault(img, []).append(cap)
    return pairs


def calculate_bleu4(reference_list, hypothesis):
    hypothesis = hypothesis.replace("।", "")
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    list_of_refs_tokens = [nltk.word_tokenize(ref.replace("।", "")) for ref in reference_list]
    smoothing = SmoothingFunction().method4
    return sentence_bleu(list_of_refs_tokens, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)


def main():
    print("Loading captions...")
    refs = load_multi_pairs(CAPTION_TXT)
    accc = load_pairs(ACCC_TXT)
    accc_mod = load_pairs(ACCC_MODIFIED_TXT)
    print(f"  caption.txt: {len(refs)} images")
    print(f"  ACCC.txt: {len(accc)} images")
    print(f"  ACCC_modified.txt: {len(accc_mod)} images")

    all_imgs = list(refs.keys())
    random.shuffle(all_imgs)
    replace_set = set(all_imgs[: int(len(all_imgs) * RATIO)])

    replaced_ref_match = 0
    replaced_mod = 0
    kept_original = 0
    skipped = 0

    for img, caps in tqdm(refs.items(), desc="Checking"):
        if img not in replace_set:
            kept_original += 1
            continue

        gen_orig = accc.get(img)
        gen_mod = accc_mod.get(img)
        if not gen_orig or not gen_mod:
            skipped += 1
            continue

        try:
            ref_match = calculate_bleu4(caps, gen_orig)
        except Exception:
            ref_match = 0.0

        if ref_match >= 0.9999:
            shortest_idx = min(range(len(caps)), key=lambda i: len(caps[i]))
            caps[shortest_idx] = gen_mod
            replaced_ref_match += 1
            continue

        try:
            mod_diff = calculate_bleu4([gen_orig], gen_mod)
        except Exception:
            mod_diff = 0.0

        if mod_diff >= 0.9999:
            kept_original += 1
        else:
            shortest_idx = min(range(len(caps)), key=lambda i: len(caps[i]))
            caps[shortest_idx] = gen_mod
            replaced_mod += 1

    output_lines = []
    for img, caps in refs.items():
        for cap in caps:
            output_lines.append(f"{img} {cap}")

    tmp_out = OUTPUT_TXT + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    os.replace(tmp_out, OUTPUT_TXT)

    total = len(refs)
    processed = kept_original + replaced_ref_match + replaced_mod
    print(f"\nTotal images: {total}")
    print(f"caption.txt vs ACCC.txt BLEU-4 = 1.0 (replaced with ACCC_modified): {replaced_ref_match} ({replaced_ref_match / max(processed, 1) * 100:.1f}%)")
    print(f"ACCC_mod vs ACCC BLEU-4 = 1.0 (kept original): {kept_original} ({kept_original / max(processed, 1) * 100:.1f}%)")
    print(f"ACCC_mod vs ACCC BLEU-4 < 1.0 (replaced with modified): {replaced_mod} ({replaced_mod / max(processed, 1) * 100:.1f}%)")
    print(f"Skipped (missing in ACCC): {skipped}")
    print(f"Output saved to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
