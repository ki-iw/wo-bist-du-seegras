import torch


class Evaluator:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def calculate_accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
        if labels.size(0) != predictions.size(0):
            raise ValueError("Tensor size mismatch!")  # noqa: TRY003

        correct = (labels == predictions).sum().item()
        total = labels.size(0)
        return correct / total
