import torch
from typing import List, Dict, Any

class MultiReferenceEvalCollator:
    """Collator that groups 5 reference Bengali captions per image for evaluation."""
    def __call__(self, batch: List[Dict[str, Any]]):
        images = torch.stack([item['image'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        references = [item['reference_captions'] for item in batch]
        return {
            'images': images,
            'image_ids': image_ids,
            'references': references
        }
