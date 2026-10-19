from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "best_brain_tumor_model.h5"
CLASS_LABELS_PATH = MODELS_DIR / "class_labels.json"

IMG_SIZE = (224, 224)
RANDOM_STATE = 42
VALID_EXT = (".jpeg", ".jpg", ".bmp", ".png")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
