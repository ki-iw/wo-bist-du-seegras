import os

import pytest
import torch
import torch.nn as nn

from zug_seegras.core.models.bag_of_seagrass import BagOfSeagrass, SeabagEnsemble

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

# TODO: move weights to shared folder on KI-IW remote or load them dynamically
MODEL_DIR = "/home/jupyter-nikolailorenz/bag-of-seagrass/Models"
SEAFEATS_WEIGHTS = os.path.join(MODEL_DIR, "SeaFeats.pt")
SEACLIP_WEIGHTS = os.path.join(MODEL_DIR, "SeaCLIP.pt")


@pytest.fixture
def bag_of_seagrass_fixture():
    return BagOfSeagrass(stride=16)


@pytest.fixture
def bag_of_seagrass_binary_fixture():
    return BagOfSeagrass(stride=16, n_classes=2)


@pytest.fixture
def seafeats_model(bag_of_seagrass_fixture):
    model = bag_of_seagrass_fixture.get_seafeats(weights_path=SEAFEATS_WEIGHTS)
    model.eval()
    return model


@pytest.fixture
def seaclips_model(bag_of_seagrass_fixture):
    model = bag_of_seagrass_fixture.get_seaclips(weights_path=SEACLIP_WEIGHTS)
    model.eval()
    return model


@pytest.fixture
def seabag_ensemble(seafeats_model, seaclips_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = SeabagEnsemble(model_1=seafeats_model, model_2=seaclips_model, device=device)
    ensemble.eval()
    return ensemble


@pytest.fixture
def seafeats_binary_model(bag_of_seagrass_binary_fixture):
    model = bag_of_seagrass_binary_fixture.get_seafeats(weights_path=SEAFEATS_WEIGHTS)
    model.eval()
    return model


@pytest.fixture
def seaclips_binary_model(bag_of_seagrass_binary_fixture):
    model = bag_of_seagrass_binary_fixture.get_seaclips(weights_path=SEACLIP_WEIGHTS)
    model.eval()
    return model


@pytest.fixture
def seabag_binary_ensemble(seafeats_binary_model, seaclips_binary_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = SeabagEnsemble(model_1=seafeats_binary_model, model_2=seaclips_binary_model, device=device)
    ensemble.eval()
    return ensemble


def test_get_seafeats_initialization(seafeats_model):
    assert isinstance(seafeats_model, nn.Module)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
def test_get_seafeats_forward_pass(seafeats_model, batch_size):
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    want = (batch_size, 4)
    got = seafeats_model(input_tensor).shape
    assert got == want


def test_get_seaclips_initialization(seaclips_model):
    assert isinstance(seaclips_model, nn.Module)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
def test_get_seaclips_forward_pass(seaclips_model, batch_size):
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    want = (batch_size, 4)
    got = seaclips_model(input_tensor).shape
    assert got == want


def test_seabag_ensemble_initialization(seabag_ensemble):
    assert isinstance(seabag_ensemble, nn.Module)


@pytest.mark.parametrize(
    "logits, want",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[-1.3363, -0.8018, -0.2673], [0.2673, 0.8018, 1.3363]]),
        ),
        (
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ),
        (
            torch.tensor([[-2.0, 0.0, 2.0], [-3.0, -1.0, 1.0]]),
            torch.tensor([[-0.8018, 0.2673, 1.3363], [-1.3363, -0.2673, 0.8018]]),
        ),
        (
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                ]
            ),
            torch.tensor(
                [
                    [[-0.8935, -0.2978, 0.2978], [0.8935, 1.4892, 2.0849]],
                    [[-0.8935, -0.8935, -0.8935], [-0.2978, -0.2978, -0.2978]],
                ]
            ),
        ),
    ],
)
def test_normalize_logits(logits, want):
    result = SeabagEnsemble.normalize_logits(logits)
    assert torch.allclose(result, want, atol=1e-4), f"Expected {want}, got {result}"
    assert result.shape == want.shape


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
def test_seabag_ensemble_forward_pass(seafeats_model, seaclips_model, seabag_ensemble, batch_size):
    device = next(seabag_ensemble.parameters()).device
    input_tensor = torch.randn(batch_size, 3, 512, 512).to(device)

    output_1 = seafeats_model(input_tensor)
    output_2 = seaclips_model(input_tensor)

    normalized_output_1 = seabag_ensemble.normalize_logits(output_1)
    normalized_output_2 = seabag_ensemble.normalize_logits(output_2)

    want_output = nn.Softmax(dim=1)((normalized_output_1 + normalized_output_2) / 2)
    want_size = (batch_size, 4)

    got = seabag_ensemble(input_tensor)

    assert got.shape == want_size
    assert torch.allclose(got, want_output, atol=1e-3)


@pytest.mark.parametrize(
    "logits, want",
    [
        (torch.tensor([[0.7, 0.1, 0.2, 0.3]]), torch.tensor([[0.7, 0.3]])),
        (torch.tensor([[0.1, 0.9, 0.2, 0.4]]), torch.tensor([[0.1, 0.9]])),
        (torch.tensor([[0.5, 0.3, 0.2, 0.5]]), torch.tensor([[0.5, 0.5]])),
        (torch.tensor([[0.5, 0.5, 0.5, 0.5]]), torch.tensor([[0.5, 0.5]])),
        (torch.tensor([[0.7, 0.1, 0.2, 0.3], [0.1, 0.9, 0.2, 0.4]]), torch.tensor([[0.7, 0.3], [0.1, 0.9]])),
    ],
)
def test_binary_classifier_happy_case(logits, want):
    got = BagOfSeagrass(n_classes=2)._binary_classifier(logits)
    assert torch.equal(got, want)


def test_binary_classifier_unhappy_case_n_classes(logits=torch.tensor([[0.7, 0.1, 0.2, 0.3]])):
    with pytest.raises(ValueError):
        BagOfSeagrass(n_classes=3)._binary_classifier(logits)


def test_binary_classifier_unhappy_case_logit_length(logits=torch.tensor([[0.7, 0.1, 0.2, 0.3, 0.0]])):
    with pytest.raises(ValueError):
        BagOfSeagrass(n_classes=2)._binary_classifier(logits)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
def test_seafeats_binary_forward_pass(seafeats_binary_model, batch_size):
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    want = (batch_size, 2)
    got = seafeats_binary_model(input_tensor).shape
    assert got == want


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
def test_seaclips_binary_forward_pass(seaclips_binary_model, batch_size):
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    want = (batch_size, 2)
    got = seaclips_binary_model(input_tensor).shape
    assert got == want


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4])
def test_seabag_binary_ensemble_forward_pass(seabag_binary_ensemble, batch_size):
    device = next(seabag_binary_ensemble.parameters()).device
    input_tensor = torch.randn(batch_size, 3, 512, 512).to(device)
    want = (batch_size, 2)

    got = seabag_binary_ensemble(input_tensor)
    assert got.shape == want
