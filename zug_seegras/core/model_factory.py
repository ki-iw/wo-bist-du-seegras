from typing import ClassVar, Optional

import torch
import torch.nn as nn

from zug_seegras.core.models.bag_of_seagrass import SeabagEnsemble, SeaCLIPModel, SeaFeatsModel
from zug_seegras.core.models.binary_resnet import BinaryResNet18
from zug_seegras.logger import getLogger

log = getLogger(__name__)


class ModelFactory:
    MODEL_REGISTRY: ClassVar = {
        "seafeats": SeaFeatsModel,
        "seaclips": SeaCLIPModel,
        "resnet18": BinaryResNet18,
    }

    def __init__(self, device: Optional[torch.device] = None):  # noqa: UP007
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self, model_name: str, n_classes: int = 2) -> nn.Module:
        model_name = model_name.lower()

        if model_name in self.MODEL_REGISTRY:
            model_class = self.MODEL_REGISTRY[model_name]
            model = model_class(n_classes=n_classes)
        elif model_name == "seabag_ensemble":
            sea_feats = SeaFeatsModel(n_classes=n_classes)
            sea_clip = SeaCLIPModel(n_classes=n_classes)
            model = SeabagEnsemble(sea_feats, sea_clip, self.device)
        else:
            raise ValueError

        model.to(self.device)
        return model

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,  # noqa: UP007
        checkpoint_path: str = "checkpoint.pth",
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,  # noqa: UP007
    ):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return checkpoint  # noqa: TRY300

        except Exception as e:
            raise ValueError from e
