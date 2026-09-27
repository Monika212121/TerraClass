import json
import torch
import shutil
import tempfile
from PIL import Image
from pathlib import Path

from .model import build_model
from .train import train_model
from .preprocess import get_eval_transform

REQUIRED_FILES = ("model.pt", "model_config.json", "class_mapping.json")



class ImageClassifier():

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_path = model_dir / 'model.pt'

        # If running this service for first time, the model is not trained
        # So first train the classifier model and save the weights in artifacts folder.
        if not self._artifacts_are_valid():
            self._train_and_install_artifacts()


        with open(model_dir / 'model_config.json', 'r') as f:
            self.config = json.load(f)

        with open(model_dir / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)

        self.idx_to_class = {int(k): v for k,v in mapping['idx_to_class'].items()}
        self.num_classes = len(self.idx_to_class)

        self.model = build_model(num_classes= self.num_classes, pretrained = False)

        self.model.load_state_dict(torch.load(model_dir/'model.pt', map_location = 'cpu'))

        self.model.eval()

        self.transform = get_eval_transform(self.config['img_size'])



    def _artifacts_are_valid(self) -> bool:
        """True only if every required file exists AND model.pt actually loads
        as a valid state dict. Catches: missing files, empty/truncated files
        from an interrupted first run, and corrupted checkpoints."""

        if not self.model_dir.exists():
            return False

        for fname in REQUIRED_FILES:
            path = self.model_dir / fname
            if not path.exists() or path.stat().st_size == 0:
                return False

        try:
            torch.load(self.model_dir / "model.pt", map_location= "cpu")
        
        except Exception:
            # Corrupt/truncated checkpoint -- e.g. process killed mid-write
            return False

        try:
            with open(self.model_dir / "model_config.json") as f:
                json.load(f)
            with open(self.model_dir / "class_mapping.json") as f:
                json.load(f)
        
        except (json.JSONDecodeError, OSError):
            return False

        return True



    def _train_and_install_artifacts(self):
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)  # clear any partial/corrupt leftovers

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            print("tmp_path: ", tmp_path)
            train_model()                  # writes REQUIRED_FILES here first

            self.model_dir.mkdir(parents=True, exist_ok=True)
            for fname in REQUIRED_FILES:
                shutil.move(str(tmp_path / fname), str(self.model_dir / fname))



    def predict(self, image: Image.Image) -> dict:
        image = image.convert('RGB')

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(tensor)

            probablitites = torch.softmax(outputs, dim= 1)[0]
            
        predicted_index = int(probablitites.argmax().item())

        confidence = float(probablitites[predicted_index].item())

        class_probs = {self.idx_to_class[i]: round(float(probablitites[i].item()), 4) for i in range(self.num_classes)}

        predictions = {
            "predicted_label": self.idx_to_class[predicted_index],
            "confidence": round(confidence, 4),
            'class_probs': class_probs,
        }

        return predictions