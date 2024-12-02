import pytest
import torch
import torch.nn as nn

from zug_seegras.core.models.binary_resnet import BinaryResNet18


@pytest.fixture
def resnet18_fixture():
    return BinaryResNet18(False, 2)


def test_resnet18_initialization(resnet18_fixture):
    model = resnet18_fixture.get_model()
    assert isinstance(model, nn.Module)


def test_get_resnet18_forward_pass(resnet18_fixture):
    input_tensor = torch.randn(1, 3, 512, 512)
    want = (1, 2)
    model = resnet18_fixture.get_model()
    got = model(input_tensor).shape
    assert got == want
