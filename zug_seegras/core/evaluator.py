from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics.functional import multiclass_f1_score
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import SeegrasDataset
from zug_seegras.core.model_factory import ModelFactory


class Evaluator:
    def __init__(
        self,
        model: Optional[nn.Module] = None,  # noqa: UP007
        model_name: Optional[str] = None,  # noqa: UP007
        checkpoint_path: Optional[str] = None,  # noqa: UP007
        n_classes: int = 2,
        transforms=None,
        device: Optional[torch.device] = None,  # noqa: UP007
    ) -> None:
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms

        model_factory = None

        if model is not None:
            self.model = model
        elif model_name is not None:
            model_factory = ModelFactory(self.device)
            self.model = model_factory.create_model(model_name=model_name, n_classes=n_classes)

        if model_factory and checkpoint_path:
            model_factory.load_checkpoint(self.model, checkpoint_path)

    def _prepare_dataloader(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        dataset.transform = self.transforms
        return DataLoader(dataset, batch_size, shuffle)

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

    def run_evaluation(
        self,
        model: Optional[nn.Module] = None,  # noqa: UP007
        dataloader: Optional[DataLoader] = None,  # noqa: UP007
        dataset: Optional[Dataset] = None,  # noqa: UP007
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> dict:
        if not dataloader:
            if not dataset:
                raise ValueError("Either 'dataloader' or 'dataset' must be provided!")  # noqa: TRY003
            dataloader = self._prepare_dataloader(dataset, batch_size, shuffle)

        if model is None:
            model = self.model

        all_labels = []
        all_predictions = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)

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
    accuracy, f1_score = evaluator.run_evaluation(dataset=dataset, batch_size=1, shuffle=False)
    print(f"Acc: {accuracy}, F1: {f1_score}")
