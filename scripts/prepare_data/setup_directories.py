import os
import shutil
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parents[1]

DATA_ROOT = PROJECT_ROOT / "FracAtlas"
IMAGES_ROOT = DATA_ROOT / "images"
LABELS_ROOT = DATA_ROOT / "Annotations" / "YOLO SEG"
SPLITS_DIR = DATA_ROOT / "Utilities" / "Overall Split"
DATASET_CSV = DATA_ROOT / "dataset.csv"

OUTPUT_SEG_DIR = PROJECT_ROOT / "FracAtlas_Segmentor_Dataset"
OUTPUT_CLS_DIR = PROJECT_ROOT / "FracAtlas_Classifier_Dataset"

def setup_directories():
    print(f"Project Root: {PROJECT_ROOT}")

    print(f"Loading splits and metadata")
    splits = {
        'train': pd.read_csv(SPLITS_DIR / "train.csv"),
        'val': pd.read_csv(SPLITS_DIR / "valid.csv"),
        'test': pd.read_csv(SPLITS_DIR / "test.csv")
    }

    full_df = pd.read_csv(DATASET_CSV)

    # Maps filename to a binary value (0: non-fractured, 1: fractured)
    fracture_map = {row['image_id']: row['fractured'] for _, row in full_df.iterrows()}

    if OUTPUT_SEG_DIR.exists():
        shutil.rmtree(OUTPUT_SEG_DIR)

    if OUTPUT_CLS_DIR.exists():
        shutil.rmtree(OUTPUT_CLS_DIR)
    
    for split_name, df in splits.items():
        print(f"\nProcessing {split_name} split ({len(df)} images)")

        for _, row in tqdm(df.iterrows(), total=len(df)):

            filename = row['image_id']
            stem = Path(filename).stem
            is_fractured = fracture_map.get(filename, 0)

            if is_fractured == 1:
                src_img = IMAGES_ROOT / "Fractured" / filename
            else:
                src_img = IMAGES_ROOT / "Non_fractured" / filename
            
            # Confirming is the csv file is in the appropriate storage folders
            if not src_img.exists(): continue

            # Classifier Dataset
            class_folder = "fractured" if is_fractured == 1 else "non_fractured"
            cls_dest_dir = OUTPUT_CLS_DIR / split_name / class_folder
            cls_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, cls_dest_dir / filename)

            # Segmentor Dataset (skip if non fractured image)
            if is_fractured == 1:
                seg_img_dest = OUTPUT_SEG_DIR / "images" / split_name
                seg_img_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, seg_img_dest / filename)

                src_lbl = LABELS_ROOT / f"{stem}.txt"
                seg_lbl_dest = OUTPUT_SEG_DIR / "labels" / split_name
                seg_lbl_dest.mkdir(parents=True, exist_ok=True)

                if src_lbl.exists():
                    shutil.copy(src_lbl, seg_lbl_dest / f"{stem}.txt")
                else:
                    with open(seg_lbl_dest / f"{stem}.txt", 'w') as f: pass

    print(f"\nCreating data.yaml for Segmentor")

    seg_yaml = {
        'path': str(OUTPUT_SEG_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'fractured'
        }
    }

    with open(OUTPUT_SEG_DIR / "data.yaml", 'w') as f:
        yaml.dump(seg_yaml, f, sort_keys=False)

    print("\nDatasets Created")

if __name__ ==  "__main__":
    setup_directories()
