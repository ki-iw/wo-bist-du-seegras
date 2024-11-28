import os

import pytest
import torch
import torch.nn as nn

from zug_seegras.core.bag_of_seagrass import BagOfSeagrass, SeabagEnsemble

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


@pytest.fixture
def seabag_ensemble(seafeats_model, seaclips_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SeabagEnsemble(model_1=seafeats_model, model_2=seaclips_model, device=device)


def test_get_seafeats_initialization(seafeats_model):
    assert isinstance(seafeats_model, nn.Module)


def test_get_seafeats_forward_pass(seafeats_model):
    input_tensor = torch.randn(1, 3, 512, 512)
    want = (1, 4)
    got = seafeats_model(input_tensor).shape
    assert got == want


def test_get_seaclips_initialization(seaclips_model):
    assert isinstance(seaclips_model, nn.Module)


def test_get_seaclips_forward_pass(seaclips_model):
    input_tensor = torch.randn(1, 3, 512, 512)
    want = (1, 4)
    got = seaclips_model(input_tensor).shape
    assert got == want


def test_seabag_ensemble_initialization(seabag_ensemble):
    assert isinstance(seabag_ensemble, nn.Module)


def test_seabag_ensemble_forward_pass(seabag_ensemble):
    device = next(seabag_ensemble.parameters()).device
    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    want = (1, 4)

    got = seabag_ensemble(input_tensor)
    assert got.shape == want
    assert torch.allclose(got.sum(dim=1), torch.tensor(1.0), atol=1e-5)
