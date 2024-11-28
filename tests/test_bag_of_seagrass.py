import os

import pytest
import torch

from zug_seegras.core.bag_of_seagrass import BagOfSeagrass

MODEL_DIR = "/home/jupyter-nikolailorenz/bag-of-seagrass/Models"
SEAFEATS_WEIGHTS = os.path.join(MODEL_DIR, "SeaFeats.pt")
SEACLIP_WEIGHTS = os.path.join(MODEL_DIR, "SeaCLIP.pt")


@pytest.fixture
def bag_of_seagrass_fixture():
    return BagOfSeagrass(stride=16)


@pytest.fixture
def seafeats_model(bag_of_seagrass_fixture):
    return bag_of_seagrass_fixture.get_seafeats(weights_path=SEAFEATS_WEIGHTS)


@pytest.fixture
def seaclips_model(bag_of_seagrass_fixture):
    return bag_of_seagrass_fixture.get_seaclips(weights_path=SEACLIP_WEIGHTS)


def test_get_seafeats_initialization(seafeats_model):
    assert isinstance(seafeats_model, torch.nn.Module)


def test_get_seafeats_forward_pass(seafeats_model):
    input_tensor = torch.randn(1, 3, 512, 512)
    want = (1, 4)
    got = seafeats_model(input_tensor).shape
    assert got == want


def test_get_seaclips_initialization(seaclips_model):
    assert isinstance(seaclips_model, torch.nn.Module)


def test_get_seaclips_forward_pass(seaclips_model):
    input_tensor = torch.randn(1, 3, 512, 512)
    want = (1, 4)
    got = seaclips_model(input_tensor).shape
    assert got == want
