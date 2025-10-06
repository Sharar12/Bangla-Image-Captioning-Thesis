import json
from collections import defaultdict

def analyze_error_cases(results_file: str):
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    length_buckets = defaultdict(list)
    for item in data.get('samples', []):
        ref_len = len(item['reference'].split())
        length_buckets[ref_len // 5 * 5].append(item['score'])

    print("Length-stratified Performance Analysis:")
    for bucket in sorted(length_buckets.keys()):
        scores = length_buckets[bucket]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  Word count [{bucket}-{bucket+4}]: avg BLEU = {avg:.2f} ({len(scores)} samples)")
