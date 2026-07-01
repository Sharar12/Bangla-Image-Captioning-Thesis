---
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  I accept the terms and conditions: checkbox
  geo: ip_location
  ? By clicking Submit below I accept the terms of the license and acknowledge that
    the information I provide will be collected stored processed and shared in accordance
    with the Meta Privacy Policy
  : checkbox
extra_gated_description: The information you provide will be collected, stored, processed
  and shared in accordance with the [Meta Privacy Policy](https://www.facebook.com/privacy/policy/).
extra_gated_button_content: Submit
extra_gated_heading: Please be sure to provide your full legal name, date of birth,
  and full organization name with all corporate identifiers. Avoid the use of acronyms
  and special characters. Failure to follow these instructions may prevent you from
  accessing this model and others on Hugging Face. You will not have the ability to
  edit this form after submission, so please ensure all information is accurate.
language:
- en
tags:
- meta-ai
- meta-pytorch
license: fair-noncommercial-research-license
base_model: facebook/pixio-vit5b16
pipeline_tag: image-feature-extraction
library_name: transformers
---

# Model Card for Pixio

Pixio is a family of versatile self-supervised vision foundation models. Pixio produces competitive dense features by simple masked autoencoding (MAE) on 2B web-crawled images with minimal human curation.

Pixio enhances MAE pre-training framework by using a deeper decoder, masking at a larger granularity, and introducing additional class tokens.

## Model Details

As described in the [Pixio](https://arxiv.org/abs/2512.15715) paper, 5 models are provided:

- 1 ViT-5B trained from scratch,
- 4 ViT-B/L/H/1B models distilled from the ViT-5B

Each model takes an image as input and returns eight class tokens and patch tokens. These models follow a standard ViT architecture, with a patch size of 16. For a 256x256 image, this results in 8 class tokens + 256 patch tokens = 264 tokens.

The models can accept larger images provided the image shapes are multiples of the patch size (16).

### Model Description

- **Developed by:** FAIR at Meta, HKU
- **Model type:** Vision Transformer
- **License:** [FAIR Noncommercial Research License](https://github.com/facebookresearch/pixio?tab=License-1-ov-file#readme)

### Model Sources

- **Repository:** [https://github.com/facebookresearch/pixio](https://github.com/facebookresearch/pixio)
- **Paper:** [In Pursuit of Pixel Supervision for Visual Pre-training](https://arxiv.org/abs/2512.15715)

### How to use

Here is how to use this model:

```python
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/pixio-vitb16')
model = AutoModel.from_pretrained('facebook/pixio-vitb16')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
last_hidden_states_norm = outputs.last_hidden_state # 8 class tokens + patch tokens after last LayerNorm
last_hidden_states = outputs.hidden_states[-1] # 8 class tokens + patch tokens before last LayerNorm
```

## Citation

```
@article{pixio,
  title={In Pursuit of Pixel Supervision for Visual Pre-training},
  author={Yang, Lihe and Li, Shang-Wen and Li, Yang and Lei, Xinjie and Wang, Dong and Mohamed, Abdelrahman and Zhao, Hengshuang and Xu, Hu},
  journal={arXiv:2512.15715},
  year={2025}
}
```
