from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "artifacts" / "model"

MODEL_PATH = MODEL_DIR / "model.pt"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"

# Training data
DATA_DIR = ROOT_DIR / "data"
CANDIDATE_DIR = DATA_DIR / "candidate_tiles"

# Training configuration
SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-3

# CPU is intentional for this assignment.
DEVICE = "cpu"