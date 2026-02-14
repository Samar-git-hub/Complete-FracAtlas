import shutil
from pathlib import Path
from ultralytics import YOLO

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parents[1]

DATASET_DIR = PROJECT_ROOT / "FracAtlas_Segmentor_Dataset"
YAML_PATH = DATASET_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs" / "segment"

def train():

    print(f"Loading YOLOv8 Small Segmentor")
    model = YOLO('yolov8s-seg.pt')

    print(f"Starting training on {DATASET_DIR}")
    
    # Changes might be induced due to library version differences
    # Note: Install albumentations in the environment, yolo uses it automatically (the original FracAtlas implementation used this)
    # Note: The author implementation mentions '608' images in the training logs. 
    # This contradicts the distribution split of 574 training images (possibly risking data leakage)
    # This is the likely cause of the discrepancy
    results = model.train(
        data=str(YAML_PATH),
        project=str(RUNS_DIR),
        name="fracatlas_legacy",
        epochs=30,
        imgsz=600,
        batch=16,
        seed=0,
        pretrained=False,
        optimizer='SGD',
        patience=50,

        device=0,
        save=True,
        val=True,
        plots=True,
        exist_ok=True
    )

    print("Training Complete")

    # Saving the model weights (YOLO saves the best and last weight files)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_dir = Path(model.trainer.save_dir)
    best_weight_path = save_dir / "weights" / "best.pt"

    if best_weight_path.exists():
        target_path = MODELS_DIR / "segmentor_legacy_best.pt"
        shutil.copy(best_weight_path, target_path)
        print(f"Best model saved to: {target_path}")
    else:
        print("Could not find best.pt weights")

if __name__ == "__main__":
    train()