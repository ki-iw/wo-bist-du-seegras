import pytest
import torch

from zug_seegras.core.bag_of_seagrass import BagOfSeagrass


@pytest.fixture
def bag_of_seagrass_fixture():
    return BagOfSeagrass(stride=16)


def test_get_seafeats_model_initialization(bag_of_seagrass_fixture):
    model = bag_of_seagrass_fixture.get_seafeats(class_list=4)
    assert isinstance(model, torch.nn.Sequential)


def test_get_seafeats_forward_pass(bag_of_seagrass_fixture):
    model = bag_of_seagrass_fixture.get_seafeats(class_list=4)
    input_tensor = torch.randn(1, 3, 512, 512)
    want = (1, 4)
    got = model(input_tensor).shape
    assert got == want


def test_get_seaclips_model_initialization(bag_of_seagrass_fixture):
    model = bag_of_seagrass_fixture.get_seaclips()
    assert isinstance(model, torch.nn.Sequential)


def test_get_seaclips_forward_pass(bag_of_seagrass_fixture):
    model = bag_of_seagrass_fixture.get_seaclips()
    input_tensor = torch.randn(1, 3, 512, 512)

    want = (1, 4)
    got = model(input_tensor).shape
    assert got == want
