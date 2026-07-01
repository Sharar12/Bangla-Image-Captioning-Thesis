import os
import random

RATIO = 1

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset40k"
)
CAPTION_TXT = os.path.join(DATA_DIR, "caption.txt")
CCC_TXT = os.path.join(DATA_DIR, "ACCC_modified.txt")
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
                pairs.setdefault(img, []).append(cap)
    return pairs


def main():
    refs = load_pairs(CAPTION_TXT)
    gens = load_pairs(CCC_TXT)

    replaced = 0
    kept_same = 0

    all_imgs = list(refs.keys())
    random.shuffle(all_imgs)
    replace_set = set(all_imgs[: int(len(all_imgs) * RATIO)])

    output_lines = []
    for img, caps in refs.items():
        gen_caps = gens.get(img, [])
        if gen_caps and img in replace_set:
            gen_cap = gen_caps[0]
            shortest_idx = min(range(len(caps)), key=lambda i: len(caps[i]))
            caps[shortest_idx] = gen_cap
            replaced += 1
        else:
            kept_same += 1
        for cap in caps:
            output_lines.append(f"{img} {cap}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    total = len(refs)
    print(f"Total images: {total}")
    print(f"Replaced: {replaced} ({replaced / total * 100:.1f}%)")
    print(f"Kept original: {kept_same} ({kept_same / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
