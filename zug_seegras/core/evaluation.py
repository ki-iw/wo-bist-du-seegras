import torch
from torcheval.metrics.functional import multiclass_f1_score


class Evaluator:
    def __init__(self, model_name: str, weights_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.get_model(model_name, self.device)
        self._load_state_dict(weights_path, self.device)

    def get_model(self, model_name: str, device=torch.device):
        model_name = model_name.lower()
        if model_name == "bag_of_seegrass":
            # TODO: Implement model
            model = None
        else:
            raise ValueError(f"Model '{model_name}' not supported.")  # noqa: TRY003

        model.to(device)
        model.eval()
        return model

    def _load_state_dict(self, weights_path: str, device: torch.device):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
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
