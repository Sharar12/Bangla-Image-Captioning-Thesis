# 🚀 60k Dataset Local Setup (Quick Guide)

Use this script to build the final dataset on your PC automatically.

### 📥 Step 1: Download Files
1. **Script & CSV:** Download everything inside `Hybrid_60k_Master_Dataset_Metadata`.
2. **Images:** Download `Flickr 30k Images`, `BNature Images` and Coco Dataset Images From Kaggle.

### 📂 Step 2: Organize Folders
Create a folder (e.g., `Thesis_Work`) and arrange files **exactly** like this:

```text
Thesis_Work/
│
├── local_dataset_builder.py      <-- (The Python Script)
├── hybrid_60k_master.csv         <-- (The Mapping CSV)
│
└── Source_Images/                <-- (Create this new folder)
    ├── Flickr30k/                <-- (Put Flickr images inside)
    ├── BNature/                  <-- (Put BNature images inside)
    └── COCO/                     <-- (Optional: Only if you downloaded COCO)
```

### ⚙️ Step 3: Run Script
Open terminal/cmd in this folder and run:
```bash
python local_dataset_builder.py
```

### 🔢 Step 4: Choose Option
*   **Type 1 (Web Mode):** If you ONLY have Flickr & BNature. The script will **download COCO images** from internet automatically. (Recommended)
*   **Type 2 (Local Mode):** If you already have the COCO folder downloaded.

✅ **Done!** You will get a new folder `Final_Hybrid_Dataset_60k` with all images renamed correctly.