import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_attention_weights(image_path: str, caption_words: list, attention_weights: np.ndarray, save_path: str = "attention_map.png"):
    """Plots spatial attention heatmaps aligned with generated Bengali words."""
    img = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(14, 8))

    num_words = len(caption_words)
    grid_size = int(np.ceil(np.sqrt(num_words + 1)))

    ax = fig.add_subplot(grid_size, grid_size, 1)
    ax.imshow(img)
    ax.set_title("Original Image")
    ax.axis("off")

    for i, word in enumerate(caption_words):
        ax = fig.add_subplot(grid_size, grid_size, i + 2)
        ax.imshow(img)
        att_map = attention_weights[i].reshape(7, 7)
        ax.imshow(att_map, cmap='jet', alpha=0.6, extent=(0, img.width, img.height, 0))
        ax.set_title(word, fontfamily='sans-serif', fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Attention visualization saved to {save_path}")
