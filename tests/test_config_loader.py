import os

import pytest
import yaml

from baltic_seagrass.core.config_loader import get_model_config, load_config


@pytest.fixture
def base_config():
    return {
        "dataset": {
            "name": "Seegras",
            "video_file": "data/input_video/trimmed_testvideo.mov",
            "label_dir": "data/input_label/default.json",
        },
        "evaluation": {"batch_size": 16, "shuffle": False},
        "checkpoint": {"dir": "data/model_checkpoints"},
    }


@pytest.fixture
def model_config():
    return {
        "model": {"name": "resnet18", "pretrained": True, "num_classes": 1},
        "training": {"batch_size": 32, "learning_rate": 0.001, "num_epochs": 20},
    }


@pytest.mark.parametrize("yaml_content", [({"key": "value"}), ([{"key": "value"}, {"key": "value"}])])
def test_load_config_happy_case(tmp_path, yaml_content):
    valid_file = tmp_path / "valid.yml"

    with open(valid_file, "w") as file:
        yaml.dump(yaml_content, file)

    loaded_data = load_config(str(valid_file))
    assert loaded_data == yaml_content


def test_load_config_unhappy_case_non_existing_file(tmp_path):
    non_existent_file = tmp_path / "non_existent.yml"

    assert not os.path.exists(non_existent_file)

    with pytest.raises(FileNotFoundError):
        load_config(str(non_existent_file))


def test_get_model_config(tmp_path, base_config, model_config):
    base_file = tmp_path / "base.yml"
    with open(base_file, "w") as file:
        yaml.dump(base_config, file)

    model_file = tmp_path / "resnet18.yml"
    with open(model_file, "w") as file:
        yaml.dump(model_config, file)

    got = get_model_config("resnet18", tmp_path)

    assert got["dataset"] == base_config["dataset"]
    assert got["evaluation"] == base_config["evaluation"]
    assert got["checkpoint"] == base_config["checkpoint"]
    assert got["model"] == model_config["model"]
    assert got["training"] == model_config["training"]
