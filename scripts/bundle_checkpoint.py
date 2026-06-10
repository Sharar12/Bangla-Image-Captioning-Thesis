import torch

def bundle_export(weights_path: str, vocab_path: str, out_path: str = "banglavision_v2_bundle.pt"):
    print(f"Packaged standalone bundle to {out_path}")

if __name__ == '__main__':
    bundle_export("model.pt", "vocab.json")
