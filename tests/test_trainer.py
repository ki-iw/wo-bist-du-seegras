import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from zug_seegras.core.trainer import Trainer


@pytest.fixture
def dummy_train_loader():
    x_train = torch.randn(100, 3, 512, 512)
    y_train = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(x_train, y_train)
    return DataLoader(train_dataset, batch_size=32)


@pytest.fixture
def dummy_test_loader():
    x_test = torch.randn(20, 3, 512, 512)
    y_test = torch.randint(0, 2, (20,))
    test_dataset = TensorDataset(x_test, y_test)
    return DataLoader(test_dataset, batch_size=32)


@pytest.mark.parametrize("model_config", os.listdir("zug_seegras/config"))
def test_trainer_train_happy_case(model_config, dummy_train_loader, dummy_test_loader):
    if not model_config.endswith(".yml") or model_config.endswith("base.yml"):
        pytest.skip(f"Skipping config file: {model_config}")

    model_name = model_config.split(".")[0]

    trainer = Trainer(model_name=model_name, train_loader=dummy_train_loader, test_loader=dummy_test_loader)

    trainer.config["training"]["num_epochs"] = 1
    trainer.train(n_eval=1)

    assert trainer.current_epoch == 1

    model_checkpoint_dir = os.path.join(
        trainer.config["checkpoint"]["dir"], trainer.config["model"]["model_name"], trainer.config["dataset"]["name"]
    )
    checkpoint_files = os.listdir(model_checkpoint_dir)
    assert len(checkpoint_files) > 0
