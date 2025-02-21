import tempfile

import pytest
import torch
import torch.nn as nn

from baltic_seagrass.core.model_factory import ModelFactory
from baltic_seagrass.core.models.bag_of_seagrass import SeaCLIPModel, SeaFeatsModel


@pytest.mark.parametrize(
    "model_name, n_classes",
    [("seafeats", 4), ("seaclips", 4), ("resnet18", 2), ("seabag_ensemble", 2)],
)
def test_model_initialization_happy_case_integration(model_name, n_classes):
    factory = ModelFactory()
    model = factory.create_model(model_name=model_name, n_classes=n_classes)
    assert isinstance(model, nn.Module)


def test_model_initialization_unhappy_case_invalid_model_name():
    factory = ModelFactory()
    with pytest.raises(ValueError):
        factory.create_model(model_name="invalid_model", n_classes=2)


def test_save_and_load_checkpoint_happy_case():
    factory = ModelFactory()
    model = factory.create_model("resnet18", n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.NamedTemporaryFile() as temp_file:
        factory.save_checkpoint(model, optimizer, checkpoint_path=temp_file.name)

        new_model = factory.create_model("resnet18", n_classes=2)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        factory.load_checkpoint(new_model, temp_file.name, optimizer=new_optimizer)

        for key in model.state_dict():
            assert torch.allclose(model.state_dict()[key], new_model.state_dict()[key], atol=1e-5)

        for key in optimizer.state_dict():
            assert optimizer.state_dict()[key] == new_optimizer.state_dict()[key]


def test_load_checkpoint_unhappy_case_invalid_path():
    factory = ModelFactory()
    model = factory.create_model("resnet18", n_classes=2)

    with pytest.raises(ValueError):
        factory.load_checkpoint(model, checkpoint_path="invalid_path.pth")


def test_seabag_ensemble_initialization_integration():
    factory = ModelFactory()
    model = factory.create_model("seabag_ensemble", n_classes=2)
    assert isinstance(model, nn.Module)
    assert hasattr(model, "model_1")
    assert hasattr(model, "model_2")
    assert isinstance(model.model_1, SeaFeatsModel)
    assert isinstance(model.model_2, SeaCLIPModel)
