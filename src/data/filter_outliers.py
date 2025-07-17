import os
from PIL import Image

def verify_images(image_dir, filenames):
    valid, corrupted = [], []
    for fn in filenames:
        path = os.path.join(image_dir, fn)
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(fn)
        except Exception:
            corrupted.append(fn)
    return valid, corrupted
