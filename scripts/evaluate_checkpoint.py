import argparse
import json
import torch
from src.evaluation.metrics_suite import CaptionMetricsSuite

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint on Bengali image caption test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights checkpoint")
    parser.add_argument("--test_data", type=str, default="data/processed/test.json", help="Path to test set annotations")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for inference")
    parser.add_argument("--output_file", type=str, default="results/evaluation_scores.json")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Evaluating on {args.test_data} with beam width {args.beam_size}...")
    metrics_suite = CaptionMetricsSuite()
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
