import warnings
from pathlib import Path

import torch
from groundingdino.config import GroundingDINO_SwinT_OGC
from groundingdino.util.inference import load_model, predict

from zug_seegras.utils import preprocess_image

warnings.filterwarnings("ignore", category=UserWarning, message=".*(meshgrid|use_reentrant|requires_grad=True).*")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*(torch.load.*weights_only=False|device.*deprecated|autocast.*deprecated).*",
)


class GroundingDinoClassifier:
    def __init__(
        self,
        device: str,
        prompt: str = "underwater plants . seaweed . plant",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompt = prompt
        self.model = self.load_model()

    def load_model(self):
        weights_path = Path.home() / "tmp/weights/groundingdino_swint_ogc.pth"
        return load_model(GroundingDINO_SwinT_OGC.__file__, weights_path, device=self.device)

    def to(self, device):
        # Auxiliary function to keep the rest of the repo consistent.
        pass

    def eval(self):
        # Auxiliary function to keep the rest of the repo consistent.
        pass

    def __call__(self, images: torch.Tensor):
        images = [preprocess_image(img, self.device) for img in images]

        predictions = []

        with torch.no_grad():
            for image in images:
                boxes, _, _ = predict(self.model, image, self.prompt, self.box_threshold, self.text_threshold)
                if len(boxes) > 0:
                    predictions.append(torch.tensor([0.0, 1.0]))
                else:
                    predictions.append(torch.tensor([1.0, 0.0]))

        return torch.stack(predictions).to(self.device)
