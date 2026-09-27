import time
import hashlib

from PIL import Image

from ml.predict import ImageClassifier



class ClassificationService:

    def __init__(self, model_dir, repository, model_version):

        self.classifier = ImageClassifier(model_dir)

        self.repository = repository

        self.model_version = model_version


    def classify(self, image: Image.Image, image_bytes: bytes):

        # ----------------------------------------
        # Start inference timer
        # ----------------------------------------

        start = time.perf_counter()

        # ----------------------------------------
        # Run local model inference
        #
        # ImageClassifier is responsible for
        # loading/training the model.
        # ----------------------------------------

        result = self.classifier.predict(
            image
        )

        inference_time_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------
        # Identify tile by content
        # ----------------------------------------

        tile_sha256 = hashlib.sha256(
            image_bytes
        ).hexdigest()

        # ----------------------------------------
        # Build classification result
        # ----------------------------------------

        return {
            "tile_sha256": tile_sha256,

            "predicted_label":
                result["predicted_label"],

            "confidence":
                result["confidence"],

            "class_probs":
                result["class_probs"],

            "inference_time_ms":
                inference_time_ms,
        }


    def list_predictions(self, limit: int = 100):
        return self.repository.get_all(limit= limit)