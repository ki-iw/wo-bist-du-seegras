import os

import yaml


def load_config(config_path: str):
    with open(config_path, "r") as file:  # noqa: UP015
        return yaml.safe_load(file)


def get_model_config(model_name: str, base_config_path: str = "baltic_seagrass/config/"):
    base_config = load_config(os.path.join(base_config_path, "base.yml"))

    model_config_path = os.path.join(base_config_path, f"{model_name}.yml")
    if not os.path.exists(model_config_path):
        raise ValueError(f"Model configuration for '{model_name}' not found!")  # noqa: TRY003

    model_config = load_config(model_config_path)

    base_config.update(model_config)

    return base_config
