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


class BagOfSeagrass:
    def __init__(self, stride: int = 16) -> None:
        self.stride = stride

    def get_seafeats(self, weights_path: Optional[str] = None, class_list: int = 4) -> nn.Module:  # noqa: UP007
        if weights_path is None:
            weights_path = "/home/jupyter-nikolailorenz/bag-of-seagrass/Models/SeaFeats.pt"

        seafeats = models.resnet18()
        layers = list(seafeats.children())[:-2]
        av_pool = nn.AvgPool2d((16, 16), stride=(self.stride, self.stride), padding=0)
        flatten = nn.Flatten(2, -1)
        layers.append(av_pool)
        layers.append(flatten)
        layers.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
        layers.append(nn.Linear(512, 512))
        layers.append(nn.Dropout(0.15))
        layers.append(nn.Linear(512, class_list))
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

        return nn.Sequential(*all_layers)

    def get_seaclips(self, weights_path: Optional[str] = None) -> nn.Module:  # noqa: UP007
        if weights_path is None:
            weights_path = "/home/jupyter-nikolailorenz/bag-of-seagrass/Models/SeaCLIP.pt"

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

        return nn.Sequential(*clip_model_pool)


class SeabagEnsemble(nn.Module):
    def __init__(self, model_1: nn.Module, model_2: nn.Module, device: torch.device):
        super().__init__()
        self.model_1 = model_1.to(device)
        self.model_2 = model_2.to(device)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output_1 = self.model_1(x)
        output_2 = self.model_2(x)

        output = (output_1 + output_2) / 2
        return self.softmax(output)
