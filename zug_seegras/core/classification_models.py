import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights


class BinaryResNet18:
    def __init__(self, pretrained: bool = False, n_classes: int = 2):
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.model = models.resnet18(weights=weights)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, n_classes)

    def get_model(self):
        return self.model
