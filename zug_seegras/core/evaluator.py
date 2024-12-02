from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import multiclass_f1_score
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import SeegrasDataset
from zug_seegras.core.models.bag_of_seagrass import BagOfSeagrass, SeabagEnsemble
from zug_seegras.core.models.binary_resnet import BinaryResNet18


class Evaluator:
    def __init__(
        self,
        model: Optional[nn.Module] = None,  # noqa: UP007
        model_name: Optional[str] = None,  # noqa: UP007
        weights_path: Optional[str] = None,  # noqa: UP007
        n_classes: int = 2,
        transforms=None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms

        if model is not None:
            self.model = model
        elif model_name is not None:
            self.model = self._initialize_model(model_name, weights_path, n_classes)
        else:
            raise ValueError("Either a model or a model_name must be provided.")  # noqa: TRY003

        self.model.to(self.device)
        self.model.eval()

    def _initialize_model(self, model_name: str, weights_path: Optional[str] = None, n_classes: int = 2) -> nn.Module:  # noqa: UP007
        model_name = model_name.lower()
        bag_of_seagrass = BagOfSeagrass(n_classes=n_classes)

        if model_name == "seafeats":
            return bag_of_seagrass.get_seafeats()
        elif model_name == "seaclips":
            return bag_of_seagrass.get_seaclips()
        elif model_name == "seabag_ensemble":
            seafeats_model = bag_of_seagrass.get_seafeats()
            seaclips_model = bag_of_seagrass.get_seaclips()
            return SeabagEnsemble(seafeats_model, seaclips_model, self.device)
        elif model_name == "resnet18":
            model = BinaryResNet18().get_model()
        else:
            raise ValueError(f"Model '{model_name}' not supported.")  # noqa: TRY003

        if weights_path:
            self._load_state_dict(model, weights_path)

        return model

    def _prepare_dataloader(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        dataset.transform = self.transforms
        return DataLoader(dataset, batch_size, shuffle)

    def _load_state_dict(self, model, weights_path: str):
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            return model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise ValueError(f"Failed to load model weights from '{weights_path}'") from e  # noqa: TRY003

    @staticmethod
    def calculate_accuracy(labels: torch.Tensor, predictions: torch.Tensor, device="cpu") -> float:
        labels = labels.to(device)
        predictions = predictions.to(device)

        if labels.size(0) != predictions.size(0):
            raise ValueError("Tensor size mismatch!")  # noqa: TRY003

        correct = torch.sum(labels == predictions).item()
        total = labels.size(0)
        return correct / total

    @staticmethod
    def calculate_f1_score(labels: torch.Tensor, predictions: torch.Tensor, device="cpu") -> float:
        labels = labels.to(device)
        predictions = predictions.to(device)

        n_classes = int(torch.unique(predictions).size(0))
        return multiclass_f1_score(labels, predictions, num_classes=n_classes).item()

    def run_evaluation(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False) -> dict:
        dataloader = self._prepare_dataloader(dataset, batch_size, shuffle)

        all_labels = []
        all_predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)

                all_labels.append(labels)
                all_predictions.append(predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        accuracy = self.calculate_accuracy(all_labels, all_predictions, device=self.device)
        f1_score = self.calculate_f1_score(all_labels, all_predictions, device=self.device)

        return accuracy, f1_score


if __name__ == "__main__":
    data_path = Path("data")

    video_file = data_path / "input_video" / "trimmed_testvideo.mov"
    label_json_path = data_path / "input_label" / "default.json"
    output_frames_dir = data_path / "output"

    dataset = SeegrasDataset(
        video_file=str(video_file),
        label_dir=str(label_json_path),
        output_dir=str(output_frames_dir),
    )

    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    evaluator = Evaluator(model_name="seaclips", transforms=transforms)
    results = evaluator.run_evaluation(dataset, batch_size=1, shuffle=False)
    print(f"Results: {results}")
