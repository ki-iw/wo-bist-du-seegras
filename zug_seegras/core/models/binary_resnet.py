import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights


class BinaryResNet18(nn.Module):
    def __init__(self, pretrained: bool = False, n_classes: int = 1):
        super().__init__()
        self.n_classes = n_classes
        self.weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = self._build_model()

    def _build_model(self):
        model = models.resnet18(weights=self.weights)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.n_classes)
        return model

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
