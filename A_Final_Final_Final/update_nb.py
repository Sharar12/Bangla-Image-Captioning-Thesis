import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'EXP_siglip2_Att_GRU_Ex/code/Jupyter Notebook/test.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ===== Change 1: Update tqdm import (Cell 2) =====
for cell in data['cells']:
    src = cell.get('source', [])
    src_str = ''.join(src)
    if 'from tqdm import tqdm' in src_str:
        new_src = []
        for line in src:
            if 'from tqdm import tqdm' in line:
                new_src.append('from tqdm.autonotebook import tqdm\n')
            else:
                new_src.append(line)
        cell['source'] = new_src
        print('✓ Updated tqdm import to autonotebook')
        break

# ===== Change 2: Rewrite run_evaluation() with progress output (Cell 20) =====
for cell in data['cells']:
    src = cell.get('source', [])
    src_str = ''.join(src)
    if 'def run_evaluation():' not in src_str:
        continue
    
    new_src = [
        'def run_evaluation():\n',
        '    """Full evaluation pipeline with detailed progress output."""\n',
        '    import time as _t\n',
        '    _start = _t.time()\n',
        '    print(f"{\'=\'*60}")\n',
        '    print(f"  🚀 Starting Evaluation Pipeline")\n',
        '    print(f"  Target Device: {DEVICE}")\n',
        '    print(f"{\'=\'*60}")\n',
        '\n',
        '    # ── Step 1: Load vocabulary ──\n',
        '    print(f"\n[1/7] 📖 Loading vocabulary...", end=" ", flush=True)\n',
        '    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")\n',
        '    if not os.path.exists(vocab_path):\n',
        '        print(f"❌ FAILED - not found at {vocab_path}")\n',
        '        return\n',
        '    vocab = Vocabulary.load(vocab_path)\n',
        '    print(f"✅ {vocab.count} tokens")\n',
        '\n',
        '    # ── Step 2: Load captions ──\n',
        '    print(f"[2/7] 📖 Loading captions...", end=" ", flush=True)\n',
        '    captions = load_captions(os.path.join(DATA_DIR, "captions.txt"))\n',
        '    image_names = list(captions.keys())\n',
        '    print(f"✅ {len(image_names)} images")\n',
        '\n',
        '    # ── Step 3: Create test split ──\n',
        '    print(f"[3/7] 🔀 Creating test split...", end=" ", flush=True)\n',
        '    n = len(image_names)\n',
        '    generator = torch.Generator().manual_seed(TRAIN_SEED)\n',
        '    perm = torch.randperm(n, generator=generator).tolist()\n',
        '    n_train = int(n * TRAIN_RATIO)\n',
        '    n_val = int(n * VAL_RATIO)\n',
        '    test_list = [image_names[i] for i in perm[n_train + n_val:]]\n',
        '    if not test_list:\n',
        '        print("❌ No test images!")\n',
        '        return\n',
        '    print(f"✅ {len(test_list)} test images ({(len(test_list)/n)*100:.1f}%)")\n',
        '\n',
        '    # ── Step 4: Load checkpoint ──\n',
        '    print(f"[4/7] 💾 Loading checkpoint...", end=" ", flush=True)\n',
        '    best_model_path = os.path.join(OUTPUT_DIR, "checkpoint_epoch_10_Final.pt")\n',
        '    if not os.path.exists(best_model_path):\n',
        '        print(f"❌ Not found at {best_model_path}")\n',
        '        return\n',
        '    current_module = sys.modules[__name__]\n',
        '    sys.modules["__main__"] = current_module\n',
        '    _t0 = _t.time()\n',
        '    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)\n',
        '    print(f"✅ loaded in {_t.time()-_t0:.1f}s")\n',
        '\n',
        '    # ── Step 5: Build model ──\n',
        '    _t0 = _t.time()\n',
        '    print(f"[5/7] 🧠 Building model (SigLIP2 → GRU→LSTM→BanglaGPT)...", end=" ", flush=True)\n',
        '    model = CaptionModel(vocab_size=vocab.vocab_size).to(DEVICE)\n',
        '    print(f"built in {_t.time()-_t0:.1f}s")\n',
        '    print(f"      Loading trained weights...", end=" ", flush=True)\n',
        '    _t0 = _t.time()\n',
        '    state_dict = checkpoint["model_state_dict"]\n',
        '    clean_state_dict = {}\n',
        '    for k, v in state_dict.items():\n',
        '        if k.startswith("_orig_mod."):\n',
        '            clean_state_dict[k[10:]] = v\n',
        '        else:\n',
        '            clean_state_dict[k] = v\n',
        '    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)\n',
        '    if missing:\n',
        '        print(f" ({len(missing)} missing)", end=" ", flush=True)\n',
        '    if unexpected:\n',
        '        print(f" ({len(unexpected)} unexpected)", end=" ", flush=True)\n',
        '    print(f"✅ weights loaded in {_t.time()-_t0:.1f}s")\n',
        '    model.eval()\n',
        '    del checkpoint\n',
        '\n',
        '    # ── Step 6: Load processor ──\n',
        '    print(f"[6/7] 🖼️ Loading image processor...", end=" ", flush=True)\n',
        '    _t0 = _t.time()\n',
        '    processor_path = (\n',
        '        SIGLIP_MODEL_PATH\n',
        '        if os.path.exists(os.path.join(SIGLIP_MODEL_PATH, "preprocessor_config.json"))\n',
        '        else "google/siglip2-base-patch32-256"\n',
        '    )\n',
        '    processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)\n',
        '    print(f"✅ loaded in {_t.time()-_t0:.1f}s")\n',
        '\n',
        '    # ── Step 7: Create DataLoader ──\n',
        '    print(f"[7/7] 📦 Creating DataLoader (batch={BATCH_SIZE}, workers={NUM_WORKERS})...", end=" ", flush=True)\n',
        '    gc.collect()\n',
        '    torch.cuda.empty_cache()\n',
        '    test_dataset = BanglaCaptionDataset(test_list, captions, processor,\n',
        '                                        os.path.join(DATA_DIR, "images"))\n',
        '    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,\n',
        '                             num_workers=NUM_WORKERS, collate_fn=collate_fn,\n',
        '                             pin_memory=True if DEVICE.type == "cuda" else False)\n',
        '    print(f"✅ {len(test_dataset)} image-caption pairs")\n',
        '\n',
        '    _setup_time = _t.time() - _start\n',
        '    print(f"\n{\'=\'*60}")\n',
        '    print(f"  ✅ Setup complete in {_setup_time:.1f}s")\n',
        '    print(f"{\'=\'*60}")\n',
        '\n',
        '    # =========================================================\n',
        '    # PHASE 1: Detailed results for random sample images\n',
        '    # =========================================================\n',
        '    print(f"\n{\'=\' * 60}")\n',
        '    print(f"  📸 [PHASE 1] Evaluating {NUM_PHASE1_SAMPLES} random test images")\n',
        '    print(f"{\'=\' * 60}")\n',
        '    _t1 = _t.time()\n',
        '    phase1_images = set(random.sample(test_list, min(NUM_PHASE1_SAMPLES, len(test_list))))\n',
        '    phase1_results = []\n',
        '    phase1_scores_map = {}\n',
        '\n',
        '    with torch.inference_mode():\n',
        '        _pbar = tqdm(total=len(phase1_images), desc="Generating captions", unit="img")\n',
        '        for batch in test_loader:\n',
        '            if not batch or len(phase1_results) >= len(phase1_images):\n',
        '                _pbar.close()\n',
        '                continue\n',
        '            pixel_values, batch_caps, img_names = batch\n',
        '            valid_indices = [idx for idx, name in enumerate(img_names) if name in phase1_images]\n',
        '            if not valid_indices:\n',
        '                continue\n',
        '            sub_pixel_values = pixel_values[valid_indices].to(DEVICE)\n',
        '            gen_caps = model.generate_beam_batch(sub_pixel_values, vocab, beam_size=1)\n',
        '\n',
        '            for index, idx in enumerate(valid_indices):\n',
        '                name = img_names[idx]\n',
        '                ref_list = batch_caps[idx]\n',
        '                gen_cap = gen_caps[index].strip()\n',
        '                m = calculate_metrics(ref_list, gen_cap)\n',
        '                m["cider"] = calculate_cider(ref_list, g
