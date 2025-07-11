import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from baltic_seagrass.core.config_loader import get_model_config
from baltic_seagrass.core.evaluator import Evaluator
from baltic_seagrass.core.model_factory import ModelFactory


class Trainer:
    def __init__(
        self,
        model_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_path: Optional[str] = None,  # noqa: UP007
    ):
        self.config = get_model_config(model_name)

        if not self.config["model"]["trainable"]:
            raise ValueError("The model is not trainable!")  # noqa: TRY003

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_factory = ModelFactory(device=self.device)

        self.current_epoch = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = self.initialize_loss_function()
        self.model = self.initialize_model(checkpoint_path)
        self.optimizer = self.initialize_optimizer()
        self.best_f1_score = 0
        self.evaluator = Evaluator(device=self.device)

    def initialize_model(self, checkpoint_path: str):
        model_params = self.config["model"]

        del model_params["trainable"]

        model = self.model_factory.create_model(**model_params)

        if checkpoint_path:
            checkpoint = self.model_factory.load_checkpoint(model, checkpoint_path)
            if "epoch" in checkpoint:
                self.current_epoch = checkpoint["epoch"]

        return model

    def initialize_loss_function(self):
        loss_name = self.config.get("training").get("loss_function", "CrossEntropyLoss")
        if loss_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif loss_name == "BCEWithLogitsLoss":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "BinaryCrossEntropy":
            return nn.BCELoss()
        raise NotImplementedError(f"Loss function '{loss_name}' is not implemented.")

    def initialize_optimizer(self):
        optimizer_name = self.config.get("training").get("optimizer", "Adam")
        learning_rate = self.config["training"]["learning_rate"]
        if optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented.")

    def train(self, n_eval: int = 5):
        num_epochs = self.config.training.num_epochs
        model_name = self.config.model.model_name
        dataset_name = self.config.training_data.name
        checkpoint_dir = self.config.checkpoint.dir

        model_checkpoint_dir = os.path.join(checkpoint_dir, model_name, dataset_name)
        os.makedirs(model_checkpoint_dir, exist_ok=True)

        self.model.to(self.device)
        for epoch in tqdm(range(self.current_epoch, num_epochs), desc="Epochs", total=num_epochs - self.current_epoch):
            self.model.train()
            running_loss = 0.0

            for inputs, labels, _ in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                labels = labels.unsqueeze(1).float()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}")

            if (epoch + 1) % n_eval == 0:
                _, f1_score, _, _ = self.evaluator.run_evaluation(model=self.model, dataloader=self.test_loader)

                checkpoint_path = os.path.join(model_checkpoint_dir, f"{model_name}_{epoch + 1}.pth")
                self.model_factory.save_checkpoint(self.model, self.optimizer, checkpoint_path, epoch + 1)
                tqdm.write(f"Model checkpoint saved at {checkpoint_path}")

                if f1_score > self.best_f1_score:
                    self.best_checkpoint = checkpoint_path
                    self.best_f1_score = f1_score

                    checkpoint_path = os.path.join(model_checkpoint_dir, f"{model_name}_best-checkpoint.pth")
                    self.model_factory.save_checkpoint(self.model, self.optimizer, checkpoint_path, epoch + 1)

            self.current_epoch += 1
