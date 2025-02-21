import os
from pathlib import Path

import pytest
import yaml

from tests.utils import create_data_loader
from baltic_seagrass.core.trainer import Trainer


@pytest.fixture
def dummy_train_loader():
    return create_data_loader(batch_size=32, num_samples=100, image_size=(3, 512, 512))


@pytest.fixture
def dummy_test_loader():
    return create_data_loader(batch_size=32, num_samples=20, image_size=(3, 512, 512))


@pytest.mark.parametrize("config_path", os.listdir("baltic_seagrass/config"))
def test_trainer_train_happy_case_integration(config_path, dummy_train_loader, dummy_test_loader):
    config_path = Path(config_path)
    model_name = config_path.stem

    if model_name in ["base"]:
        pytest.skip(f"Skip config file: {config_path.name}")

    config_file_path = Path("baltic_seagrass/config") / config_path
    with open(config_file_path, "r") as config_file:  # noqa: UP015
        config = yaml.safe_load(config_file)

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
