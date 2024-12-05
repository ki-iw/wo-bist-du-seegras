from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from zug_seegras.core.trainer import Trainer


def create_data_loader(batch_size: int, num_samples: int, image_size: tuple):
    x_data = torch.randn(num_samples, *image_size)
    y_data = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def dummy_train_loader():
    return create_data_loader(batch_size=32, num_samples=100, image_size=(3, 512, 512))


@pytest.fixture
def dummy_test_loader():
    return create_data_loader(batch_size=32, num_samples=20, image_size=(3, 512, 512))


def model_configs():
    config_dir = Path("zug_seegras/config")
    return [f for f in config_dir.iterdir() if f.suffix == ".yml" and f.name != "base.yml"]


@pytest.mark.parametrize("model_config", model_configs())
def test_trainer_train_happy_case_integration(model_config, dummy_train_loader, dummy_test_loader):
    model_name = model_config.stem
    config = yaml.safe_load(model_config.read_text())

    if not config["model"]["trainable"]:
        with pytest.raises(ValueError, match="The model is not trainable"):
            trainer = Trainer(model_name=model_name, train_loader=dummy_train_loader, test_loader=dummy_test_loader)
            trainer.initialize_model(checkpoint_path=None)
        return

    trainer = Trainer(model_name=model_name, train_loader=dummy_train_loader, test_loader=dummy_test_loader)

    trainer.config["training"]["num_epochs"] = 1
    trainer.train(n_eval=1)

    assert trainer.current_epoch == 1

    model_checkpoint_dir = (
        Path(trainer.config["checkpoint"]["dir"])
        / trainer.config["model"]["model_name"]
        / trainer.config["dataset"]["name"]
    )
    checkpoint_files = list(model_checkpoint_dir.iterdir())
    assert len(checkpoint_files) > 0
