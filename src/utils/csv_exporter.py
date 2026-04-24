import csv

def export_evaluation_to_csv(results: dict, filepath: str):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "BLEU-1", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"])
        for m, s in results.items():
            writer.writerow([m, s.get('BLEU-1'), s.get('BLEU-4'), s.get('ROUGE-L'), s.get('METEOR'), s.get('CIDEr')])
