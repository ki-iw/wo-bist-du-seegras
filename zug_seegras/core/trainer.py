import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import SeegrasDataset
from zug_seegras.core.model_factory import ModelFactory
from zug_seegras.logger import getLogger

log = getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,  # noqa: UP007
    ):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_factory = ModelFactory(device=self.device)

        self.current_epoch = 0
        self.dataloader = self.initialize_dataloader()
        self.criterion = self.initialize_loss_function()
        self.optimizer = self.initialize_optimizer()
        self.model = self.initialize_model(checkpoint_path)

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, "r") as file:  # noqa: UP015
            return yaml.safe_load(file)

    def initialize_model(self, checkpoint_path: str):
        model_config = self.config["model"]

        model = self.model_factory.create_model(model_name=model_config["name"], n_classes=model_config["num_classes"])

        if checkpoint_path:
            checkpoint = self.model_factory.load_checkpoint(model, checkpoint_path, self.optimizer)
            if "epoch" in checkpoint:
                self.current_epoch = checkpoint["epoch"]

        return model

    def initialize_dataloader(self):
        dataset_config = self.config["dataset"]
        input_path = self.config["dataset"]["video_file"]
        label_path = self.config["dataset"]["label_dir"]
        output_path = self.config["dataset"]["output_dir"]

        transforms = Compose(
            [
                Resize((512, 512)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if dataset_config["name"] == "seegras":
            dataset = SeegrasDataset(input_path, label_path, output_path, transform=transforms)
        else:
            raise NotImplementedError(f"Dataset type '{dataset_config['name']}' is not supported.")

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=self.config["training"]["shuffle"],
        )

    def initialize_loss_function(self):
        return nn.CrossEntropyLoss()

    def initialize_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.config["training"]["learning_rate"])


def train(self):
    num_epochs = self.config["training"]["num_epochs"]
    model_name = self.config["model"]["name"]
    dataset_name = self.config["dataset"]["name"]
    checkpoint_dir = self.config["checkpoint"]["dir"]

    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name, dataset_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    self.model.to(self.device)
    for epoch in range(self.current_epoch, num_epochs):
        self.model.train()
        running_loss = 0.0

        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        log.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_checkpoint_dir, f"{model_name}_{epoch + 1}.pth")
            self.model_factory.save_checkpoint(self.model, self.optimizer, checkpoint_path, epoch + 1)
            log.debug(f"Model checkpoint saved at {checkpoint_path}")
