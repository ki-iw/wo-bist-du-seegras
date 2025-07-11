from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BaseSeagrassModel(nn.Module):
    def __init__(self, n_classes: int, stride: int):
        super().__init__()
        self.n_classes = n_classes
        self.stride = stride
        self.softmax = nn.Softmax(dim=1)

    def _binary_classifier(self, logits):
        if logits.size(1) == 4 and self.n_classes == 2:
            background_logits = logits[:, 0]
            target_logits, _ = torch.max(logits[:, 1:], dim=1)
            return torch.stack((background_logits, target_logits), dim=1)
        else:
            raise ValueError


class SeaFeatsModel(BaseSeagrassModel):
    def __init__(
        self,
        n_classes: int = 4,
        stride: int = 16,
        weights_path: Optional[str] = None,  # noqa: UP007
        finetune: bool = False,
    ):
        super().__init__(n_classes, stride)
        self.finetune = finetune
        self.model = self._build_model(weights_path)

    def _build_model(self, weights_path: Optional[str] = None) -> nn.Module:  # noqa: UP007
        if weights_path is None:
            weights_path = "/mnt/data/ZUG-Seegras/weights/bag-of-seagrass/SeaFeats.pt"

        seafeats = models.resnet18()
        layers = list(seafeats.children())[:-2]
        av_pool = nn.AvgPool2d((16, 16), stride=(self.stride, self.stride), padding=0)
        flatten = nn.Flatten(2, -1)
        layers.append(av_pool)
        layers.append(flatten)
        layers.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
        layers.append(nn.Linear(512, 512))
        layers.append(nn.Dropout(0.15))
        layers.append(nn.Linear(512, 4))
        seafeats = nn.Sequential(*layers)

        seafeats.load_state_dict(torch.load(weights_path, weights_only=True))

        layers_alter_1 = list(seafeats.children())[:8]
        layers_alter_2 = list(seafeats.children())[11:]

        all_layers = []
        for layer in layers_alter_1:
            all_layers.append(layer)

        all_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        all_layers.append(nn.Flatten(1))

        for layer in layers_alter_2:
            all_layers.append(layer)

        model = nn.Sequential(*all_layers)

        if self.finetune and self.n_classes == 2:
            model[-1] = nn.Linear(512, 1)
        return model

    def forward(self, x):
        x = self.model(x)
        if self.finetune and self.n_classes == 2:
            x = torch.sigmoid(x)
        elif not self.finetune:
            if self.n_classes == 2:
                x = self._binary_classifier(x)
            x = self.softmax(x)
        return x


class SeaCLIPModel(BaseSeagrassModel):
    def __init__(
        self,
        n_classes: int = 4,
        stride: int = 16,
        weights_path: Optional[str] = None,  # noqa: UP007
        finetune: bool = False,
    ):
        super().__init__(n_classes, stride)
        self.finetune = finetune
        self.model = self._build_model(weights_path)

    def _build_model(self, weights_path: Optional[str]):  # noqa: UP007
        if weights_path is None:
            weights_path = "/mnt/data/ZUG-Seegras/weights/bag-of-seagrass/SeaCLIP.pt"

        clip_model_load = models.resnet18()
        clip_model_load.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.15), nn.Linear(512, 4))

        clip_model_load.load_state_dict(torch.load(weights_path, weights_only=True))

        all_layers = list(clip_model_load.children())
        clip_model_pool = all_layers[:-2]
        av_pool = nn.AvgPool2d((8, 8), stride=(self.stride, self.stride), padding=0)
        flatten = nn.Flatten(2, -1)
        clip_model_pool.append(av_pool)
        clip_model_pool.append(flatten)
        clip_model_pool.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
        clip_model_pool.append(Lambda(lambda x: x.squeeze(1)))
        clip_model_pool.append(all_layers[-1])

        model = nn.Sequential(*clip_model_pool)

        if self.finetune and self.n_classes == 2:
            model[-1] = nn.Linear(512, 1)

        return model

    def forward(self, x):
        x = self.model(x)
        if self.finetune and self.n_classes == 2:
            x = torch.sigmoid(x)
        elif not self.finetune:
            if self.n_classes == 2:
                x = self._binary_classifier(x)
            x = self.softmax(x)
        return x


class SeabagEnsemble(nn.Module):
    def __init__(self, model_1: nn.Module, model_2: nn.Module, device: torch.device):
        super().__init__()
        self.model_1 = model_1.to(device)
        self.model_2 = model_2.to(device)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def normalize_logits(logits):
        mean = logits.mean()
        std = logits.std()
        normalized_logits = (logits - mean) / std
        if std == 0.0:
            return torch.zeros_like(logits)
        else:
            return normalized_logits

    def forward(self, x):
        logits_1 = self.model_1(x)
        logits_2 = self.model_2(x)

        normalized_logits_1 = self.normalize_logits(logits_1)
        normalized_logits_2 = self.normalize_logits(logits_2)

        combined_logits = (normalized_logits_1 + normalized_logits_2) / 2
        return self.softmax(combined_logits)
