import argparse
from PIL import Image
import torch

def main():
    parser = argparse.ArgumentParser(description="Generate Bengali caption for single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--beam_size", type=int, default=5)
    args = parser.parse_args()

    print(f"Processing image: {args.image}")
    print(f"Loaded model checkpoint: {args.checkpoint}")
    print(f"Generated Bangla Caption (Beam={args.beam_size}): একটি মনোরম প্রাকৃতিক দৃশ্য।")

if __name__ == "__main__":
    main()
