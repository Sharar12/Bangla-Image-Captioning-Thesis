import numpy as np
from PIL import Image

def overlay_attention_map(image: Image.Image, alpha: np.ndarray, alpha_weight: float = 0.6):
    img_resized = image.resize((224, 224))
    # Reshape alpha to 14x14 or 7x7 grid
    grid_size = int(np.sqrt(alpha.shape[0]))
    alpha_grid = alpha.reshape((grid_size, grid_size))

    alpha_resized = Image.fromarray((alpha_grid * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return alpha_resized
