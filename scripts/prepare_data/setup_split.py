import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensuring reproducibility of the train, val and test splits
VAL_SIZE = 0.12
TEST_SIZE = 0.08
RANDOM_SEED = 42

def setup_overall_split(root_dir):

    project_root = Path(root_dir)
    frac_atlas_path = project_root / "FracAtlas"

    dataset_csv_path = frac_atlas_path / "dataset.csv"
    orig_split_dir = frac_atlas_path / "Utilities" / "Fracture Split"

    output_dir = frac_atlas_path / "Utilities" / "Overall Split"

    print(f"Reading original fracture splits from: {orig_split_dir}")
    try:
        frac_train = pd.read_csv(orig_split_dir / "train.csv")
        frac_val = pd.read_csv(orig_split_dir / "valid.csv")
        frac_test = pd.read_csv(orig_split_dir / "test.csv")
    except FileNotFoundError as e:
        print("Error: Could not find original split files. Ensure FracAtlas unzipped corectly. \n{e}")
        return
    
    frac_train['split_type'] = 'train'
    frac_val['split_type'] = 'val'
    frac_test['split_test'] = 'test'

    # Excluding these from healthy bone scans, ensuring zero data leakage
    known_fractured_ids = set(pd.concat([frac_train, frac_val, frac_test])['image_id'])
    print(f"Loaded {len(known_fractured_ids)} fracture images from original splits")

    print("Processing healthy images")
    full_df = pd.read_csv(dataset_csv_path)

    healthy_df = full_df[~full_df['image_id'].isin(known_fractured_ids)].copy()
    print(f"Found {len(healthy_df)} healthy images")

    # Separating Train and Holdout (Val + Test) sets
    healthy_train, healthy_hold = train_test_split(
        healthy_df,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Separating Val and Test from the Holdout set
    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    healthy_val, healthy_test = train_test_split(
        healthy_hold,
        train_size=relative_val_size,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(f"Healthy Split created: Train={len(healthy_train)}, Val={len(healthy_val)}, Test={len(healthy_test)}")

    overall_train = pd.concat([frac_train[['image_id']], healthy_train[['image_id']]]).sample(frac=1, random_state=RANDOM_SEED)
    overall_val = pd.concat([frac_val[['image_id']], healthy_val[['image_id']]]).sample(frac=1, random_state=RANDOM_SEED)
    overall_test = pd.concat([frac_test[['image_id']], healthy_test[['image_id']]]).sample(frac=1, random_state=RANDOM_SEED)

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_train.to_csv(output_dir / "train.csv", index=False)
    overall_val.to_csv(output_dir / "valid.csv", index=False)
    overall_test.to_csv(output_dir / "test.csv", index=False)

    print(f"Overall Split saved to: {output_dir}")
    print(f"Total Train: {len(overall_train)}")
    print(f"Total Val: {len(overall_val)}")
    print(f"Total Test: {len(overall_test)}")

if __name__ == "__main__":
    
    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parent.parent
    setup_overall_split(project_root)