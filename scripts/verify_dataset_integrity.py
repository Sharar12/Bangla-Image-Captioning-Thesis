import json
import sys

def audit_annotations(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[Audit] Found {len(data)} annotations.")
    empty_captions = sum(1 for d in data if not d.get('caption'))
    print(f"[Audit] Empty captions: {empty_captions}")
    return empty_captions == 0

if __name__ == "__main__":
    print("Dataset integrity audit routine loaded.")
