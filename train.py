import argparse
import yaml
import torch
from src.utils.seed import seed_everything
from src.utils.logger import get_logger

def main():
    parser = argparse.ArgumentParser(description="Train Bangla Image Captioning Models")
    parser.add_argument("--config", type=str, default="configs/siglip2_banglagpt.yaml", help="Path to experiment config YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    logger = get_logger()
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("seed", 42))
    logger.info("Initialized training environment successfully.")

if __name__ == "__main__":
    main()
