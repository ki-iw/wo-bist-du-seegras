from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

from zug_seegras.core.fiftyone_logger import FiftyOneLogger
from zug_seegras.core.model_factory import ModelFactory


class Evaluator:
    def __init__(
        self,
        model: Optional[nn.Module] = None,  # noqa: UP007
        model_name: Optional[str] = None,  # noqa: UP007
        checkpoint_path: Optional[str] = None,  # noqa: UP007
        n_classes: int = 2,
        device: Optional[torch.device] = None,  # noqa: UP007
        save_fiftyone: bool = False,
    ) -> None:
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_fiftyone = save_fiftyone

        model_factory = None

        if model is not None:
            self.model = model
        elif model_name is not None:
            model_factory = ModelFactory(self.device)
            self.model = model_factory.create_model(model_name=model_name, n_classes=n_classes)

        if model_factory and checkpoint_path:
            model_factory.load_checkpoint(self.model, checkpoint_path)

        if self.save_fiftyone:
            self.fiftyone_logger = FiftyOneLogger()

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
    ) -> dict:
        if not dataloader:
            raise ValueError("'dataloader' must be provided!")  # noqa: TRY003

        if model is None:
            model = self.model

        all_labels = []
        all_predictions = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels, paths = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)

                all_labels.append(labels)
                all_predictions.append(predicted)

                if self.save_fiftyone:
                    self.fiftyone_logger.add_batch_samples(inputs, paths, labels, predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        accuracy = self.calculate_accuracy(all_labels, all_predictions, device=self.device)
        f1_score = self.calculate_f1_score(all_labels, all_predictions, device=self.device)

        return accuracy, f1_score
