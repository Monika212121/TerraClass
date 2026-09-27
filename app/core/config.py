from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = ROOT_DIR / "artifacts"/ "model"

DATABASE_PATH = ROOT_DIR / "storage"/ "terraclass.db"


MODEL_VERSION = "mobilenetv3-small-v1"