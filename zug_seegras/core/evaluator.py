from typing import Optional

import torch
import torch.nn as nn
import torcheval.metrics.functional as mf
from torch.utils.data import DataLoader
from tqdm import tqdm

from zug_seegras import config
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

                # binary classification
                if config.model.n_classes == 1:
                    probs = torch.sigmoid(outputs).squeeze()
                    predicted = (probs > 0.5).long()
                else:
                    # TODO check if this is still correct? We dont have multiclasses though
                    _, predicted = torch.max(outputs, 1)

                all_labels.append(labels)
                all_predictions.append(predicted)

                if self.save_fiftyone:
                    self.fiftyone_logger.add_batch_samples(inputs, paths, labels, predicted)

        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)

        # TODO infer n_classes from somewhere else
        n_classes = int(torch.unique(all_labels).size(0))
        average = "macro" if n_classes == 2 else "micro"
        accuracy = mf.multiclass_accuracy(all_labels, all_predictions, num_classes=n_classes, average=average).item()
        precision = mf.multiclass_precision(all_labels, all_predictions, num_classes=n_classes, average=average).item()
        recall = mf.multiclass_recall(all_labels, all_predictions, num_classes=n_classes, average=average).item()
        f1_score = mf.multiclass_f1_score(all_labels, all_predictions, num_classes=n_classes, average=average).item()

        tqdm.write(
            f"Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )
        return accuracy, f1_score
