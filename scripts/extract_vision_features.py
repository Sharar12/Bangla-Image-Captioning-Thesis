import torch
from tqdm import tqdm

def precompute_dataset_features(encoder, dataloader, cache_manager, device="cuda"):
    encoder.eval()
    with torch.no_grad():
        for imgs, img_ids in tqdm(dataloader, desc="Extracting visual features"):
            imgs = imgs.to(device)
            feats = encoder(imgs)
            for i, i_id in enumerate(img_ids):
                cache_manager.save(i_id, feats[i])
    print("[Extract] Feature precomputation complete.")
