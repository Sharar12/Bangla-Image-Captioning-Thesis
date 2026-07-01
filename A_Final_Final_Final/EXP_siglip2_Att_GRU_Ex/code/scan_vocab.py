import os
import sys
import json
import torch
from collections import Counter
from multiprocessing import Pool, cpu_count

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(MODEL_FOLDER, "dataset40k")
OUTPUT_DIR = os.path.join(MODEL_FOLDER, "output")

TRAIN_RATIO = 0.99
VAL_RATIO = 0.005
TEST_RATIO = 0.005
MIN_FREQ = 1
NUM_WORKERS = cpu_count()

_worker_captions = None
_worker_all_words = None
_worker_train_ratio = None
_worker_min_freq = None


def _init_worker(captions, all_words, train_ratio, min_freq):
    global _worker_captions, _worker_all_words, _worker_train_ratio, _worker_min_freq
    _worker_captions = captions
    _worker_all_words = all_words
    _worker_train_ratio = train_ratio
    _worker_min_freq = min_freq


def _eval_seed(seed):
    try:
        captions = _worker_captions
        all_words = _worker_all_words
        train_ratio = _worker_train_ratio
        min_freq = _worker_min_freq
        image_names = list(captions.keys())
        n = len(image_names)
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator).tolist()
        n_train = min(int(n * train_ratio), n)
        train_idx = perm[:n_train]
        train_list = [image_names[i] for i in train_idx]
        train_captions = {k: captions[k] for k in train_list}
        train_vocab = Vocabulary()
        train_vocab.build_from_captions(train_captions, min_freq=min_freq)
        train_words = train_vocab.get_words()
        covered = len(train_words & all_words)
        coverage = covered / len(all_words) * 100
        missing = all_words - train_words
        return seed, coverage, train_vocab.vocab_size, len(missing), None
    except Exception as e:
        return seed, 0.0, 0, 0, str(e)


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.count = 4

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1

    def build_from_captions(self, captions_dict, min_freq=2):
        freq = Counter()
        for caps in captions_dict.values():
            for cap in caps:
                for word in cap.split():
                    freq[word] += 1
        for word, count in freq.items():
            if count >= min_freq:
                self.add_word(word)

    @property
    def vocab_size(self):
        return self.count

    def get_words(self):
        return set(self.word2idx.keys())


def load_captions(caption_file):
    captions = {}
    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                fname, cap = parts[0], parts[1]
                if fname not in captions:
                    captions[fname] = []
                captions[fname].append(cap)
    return captions


def scan_ratio(captions, all_words, train_ratio, min_freq, start, end, step, prev_results):
    print(f"\n{'=' * 60}")
    print(f"Scanning train_ratio={train_ratio}, min_freq={min_freq}, seeds {start}-{end-1}  (workers={NUM_WORKERS})")
    print(f"{'=' * 60}")

    full_coverage = []
    best_coverage = 0
    best_seeds = []

    ratio_key = f"ratio_{train_ratio}_minfreq_{min_freq}"
    saved = prev_results.get(ratio_key, {})
    newly_scanned = 0

    seeds_to_scan = []
    for seed in range(start, end, step):
        sk = str(seed)
        if sk in saved:
            cov = saved[sk]["coverage"]
            if cov == 100.0:
                full_coverage.append(seed)
            if cov > best_coverage:
                best_coverage = cov
                best_seeds = [seed]
            elif cov == best_coverage:
                best_seeds.append(seed)
        else:
            seeds_to_scan.append(seed)

    if seeds_to_scan:
        batch_size = max(NUM_WORKERS * 4, 100)
        try:
            with Pool(
                NUM_WORKERS,
                initializer=_init_worker,
                initargs=(captions, all_words, train_ratio, min_freq),
            ) as pool:
                found_100 = False
                for i in range(0, len(seeds_to_scan), batch_size):
                    if found_100:
                        break
                    batch = seeds_to_scan[i : i + batch_size]
                    for result in pool.imap_unordered(_eval_seed, batch):
                        seed, coverage, vocab_size, missing_count, err = result
                        sk = str(seed)
                        if err:
                            print(f"  Seed {seed}: ERROR — {err}")
                            saved[sk] = {
                                "coverage": 0.0,
                                "vocab_size": 0,
                                "missing": 0,
                                "error": err,
                            }
                            newly_scanned += 1
                            continue

                        saved[sk] = {
                            "coverage": coverage,
                            "vocab_size": vocab_size,
                            "missing": missing_count,
                        }
                        newly_scanned += 1

                        if coverage == 100.0:
                            full_coverage.append(seed)
                            print(f"  Seed {seed}: 100% coverage ✓")
                            pool.terminate()
                            found_100 = True
                            break
                        elif coverage > best_coverage:
                            best_coverage = coverage
                            best_seeds = [seed]
                            print(
                                f"  Seed {seed}: {coverage:.2f}% (new best, missing {missing_count})"
                            )
                        elif coverage == best_coverage:
                            best_seeds.append(seed)
                        else:
                            print(f"  Seed {seed}: {coverage:.2f}% (missing {missing_count})")

                    prev_results[ratio_key] = saved
                    save_results(prev_results)
                    print(f"  [{i + len(batch)}/{len(seeds_to_scan)} scanned]\n")
        except KeyboardInterrupt:
            print(f"\n  Interrupted! Saving {newly_scanned} scanned results...")
            prev_results[ratio_key] = saved
            save_results(prev_results)
            print("  Progress saved. Exiting.")
            sys.exit(130)

    prev_results[ratio_key] = saved
    save_results(prev_results)

    total_seeds = (end - start) // step
    print(f"\n--- Results for train_ratio={train_ratio} ---")
    print(f"Seeds scanned: {newly_scanned} new + {len(saved) - newly_scanned} cached (total {len(saved)}/{total_seeds})")
    if full_coverage:
        print(f"Seeds with 100% coverage ({len(full_coverage)} total):")
        for s in full_coverage[:50]:
            print(f"  {s}")
        if len(full_coverage) > 50:
            print(f"  ... and {len(full_coverage) - 50} more")
    else:
        print(f"Best coverage: {best_coverage:.2f}% at seeds: {best_seeds[:20]}")

    return full_coverage, best_seeds[0] if best_seeds else None


TRAINSCCC_PATH = os.path.join(CURRENT_DIR, "trainsCCC.py")


def generate_oov_report(seed, ratio, min_freq, captions):
    """
    Generate a .txt report listing val/test images whose captions contain
    words not present in the training vocabulary.
    """
    image_names = list(captions.keys())
    n = len(image_names)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    n_train = int(n * ratio)
    n_val = int(n * VAL_RATIO)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_list = [image_names[i] for i in train_idx]
    val_list = [image_names[i] for i in val_idx]
    test_list = [image_names[i] for i in test_idx]

    train_captions = {k: captions[k] for k in train_list}
    vocab = Vocabulary()
    vocab.build_from_captions(train_captions, min_freq=min_freq)
    known = vocab.get_words()

    report_path = os.path.join(OUTPUT_DIR, f"oov_report_ratio{ratio}_seed{seed}_minfreq{min_freq}.txt")
    total_oov_images = 0
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"TRAIN: {len(train_list)} images\n")
        f.write(f"VAL:   {len(val_list)} images\n")
        f.write(f"TEST:  {len(test_list)} images\n\n")
        for split_name, split_list in [("VAL", val_list), ("TEST", test_list)]:
            f.write(f"=== {split_name} ({len(split_list)} images) ===\n")
            count = 0
            for img_name in split_list:
                oov_words = set()
                for cap in captions.get(img_name, []):
                    for w in cap.split():
                        if w not in known:
                            oov_words.add(w)
                if oov_words:
                    count += 1
                    f.write(f"{img_name}  {', '.join(sorted(oov_words))}\n")
            f.write(f"Total {split_name} images with OOV words: {count}\n\n")
            total_oov_images += count

    print(f"\nOOV report saved to {report_path}")
    print(f"  Total images with unknown words: {total_oov_images}")
    return report_path


def save_results(results):
    result_file = os.path.join(OUTPUT_DIR, "vocab_scan_results.json")
    tmp_file = result_file + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, result_file)
    except Exception as e:
        print(f"  Warning: failed to save results — {e}")


def load_prev_results():
    result_file = os.path.join(OUTPUT_DIR, "vocab_scan_results.json")
    if os.path.exists(result_file):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def apply_to_trainsccc(ratio, seed):
    if not os.path.exists(TRAINSCCC_PATH):
        print(f"Error: {TRAINSCCC_PATH} not found")
        return

    with open(TRAINSCCC_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.lstrip().startswith("TRAIN_RATIO "):
            new_lines.append(f"TRAIN_RATIO = {ratio}")
        elif line.lstrip().startswith("TRAIN_SEED "):
            new_lines.append(f"TRAIN_SEED = {seed}")
        else:
            new_lines.append(line)

    with open(TRAINSCCC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    print(f"\nApplied TRAIN_RATIO={ratio}, TRAIN_SEED={seed} to trainsCCC.py")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    args = sys.argv[1:]

    if "--apply" in args:
        idx = args.index("--apply")
        if len(args) > idx + 2:
            ratio = float(args[idx + 1])
            seed = int(args[idx + 2])
            apply_to_trainsccc(ratio, seed)
            sys.exit(0)
        else:
            print("Usage: --apply RATIO SEED")
            sys.exit(1)

    min_freq = MIN_FREQ
    train_ratios = []
    seed_range_args = []
    for a in args:
        if a.startswith("--min-freq="):
            min_freq = int(a.split("=")[1])
            continue
        try:
            v = float(a)
            if 0 < v < 1:
                train_ratios.append(v)
            else:
                seed_range_args.append(str(int(v)))
        except ValueError:
            pass

    if not train_ratios:
        train_ratios = [TRAIN_RATIO]

    start = 0
    end = 10000
    step = 1

    if len(seed_range_args) >= 1:
        start = int(seed_range_args[0])
    if len(seed_range_args) >= 2:
        end = int(seed_range_args[1])
    if len(seed_range_args) >= 3:
        step = int(seed_range_args[2])

    print("Loading all captions...")
    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))
    print(f"Total images: {len(captions)}")

    print("Building full vocabulary from ALL captions...")
    full_vocab = Vocabulary()
    full_vocab.build_from_captions(captions, min_freq=min_freq)
    all_words = full_vocab.get_words()
    print(f"Full vocabulary size (min_freq={min_freq}): {full_vocab.vocab_size}")

    prev_results = load_prev_results()
    if prev_results:
        print(f"Loaded cached results")

    all_results = {}
    best_seed_by_ratio = {}
    try:
        for tr in train_ratios:
            full_cov, best_seed = scan_ratio(captions, all_words, tr, min_freq, start, end, step, prev_results)
            all_results[tr] = full_cov
            best_seed_by_ratio[tr] = best_seed
    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        save_results(prev_results)
        print("Progress saved. Exiting.")
        sys.exit(130)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY (min_freq={min_freq}) — apply with: python scan_vocab.py --apply RATIO SEED")
    print(f"{'=' * 60}")
    for tr, seeds in all_results.items():
        if seeds:
            print(f"  ratio={tr}: {len(seeds)} seeds with 100% coverage")
            print(f"    Example: python scan_vocab.py --apply {tr} {seeds[0]}")
        else:
            print(f"  ratio={tr}: no 100% coverage seed found in range")

    for tr in train_ratios:
        seed = all_results[tr][0] if all_results[tr] else best_seed_by_ratio[tr]
        if seed is not None:
            generate_oov_report(seed, tr, min_freq, captions)
            break
