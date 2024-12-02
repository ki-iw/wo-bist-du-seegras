import tempfile

import pytest
import torch
import torch.nn as nn

from zug_seegras.core.classification_models import BinaryResNet18
from zug_seegras.core.evaluation import Evaluator as e


@pytest.fixture
def binary_resnet18_fixture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return BinaryResNet18(pretrained=True).get_model().to(device)


@pytest.mark.parametrize(
    "labels, predictions, want",
    [  # Binary
        (torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]), 1.0),
        (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 0, 1]), 0.5),
        (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 1, 0]), 0.0),
        # Multi-class
        (torch.tensor([0, 1, 2, 3]), torch.tensor([0, 2, 2, 3]), 0.75),
        (torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]), 1.0),
        (torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0]), 0.0),
    ],
)
def test_calculate_accuracy_happy_case(labels, predictions, want):
    got = e.calculate_accuracy(labels, predictions)
    assert got == want
    assert isinstance(got, float)


@pytest.mark.parametrize(
    "labels, predictions",
    [
        (torch.tensor([0, 1]), torch.tensor([0])),
        (torch.tensor([0, 1]), torch.tensor([0, 0, 0])),
    ],
)
def test_calculate_accuracy_unhappy_case_tensor_size(labels, predictions):
    with pytest.raises(ValueError):
        e.calculate_accuracy(labels, predictions)


def test_calculate_accuracy_happy_case_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    labels = torch.tensor([0, 1, 1, 0], device="cpu")
    predictions = torch.tensor([0, 1, 0, 0], device="cuda")

    try:
        e.calculate_accuracy(labels, predictions)
    except RuntimeError:
        pytest.fail("Device handling error!")


def test_calculate_f1_score_happy_case(
    labels=torch.tensor([0, 0, 1, 1, 1]), predictions=torch.tensor([0, 0, 0, 0, 1]), want=0.6
):
    got = e.calculate_f1_score(labels, predictions)
    assert got == pytest.approx(want, rel=1e-3)
    assert isinstance(got, float)


@pytest.mark.parametrize(
    "labels, predictions",
    [
        (torch.tensor([0, 1]), torch.tensor([0])),
        (torch.tensor([0, 1]), torch.tensor([0, 0, 0])),
    ],
)
def test_calculate_f1_score_unhappy_case_tensor_size(labels, predictions):
    with pytest.raises(ValueError):
        e.calculate_f1_score(labels, predictions)


def test_calculate_f1_score_happy_case_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    labels = torch.tensor([0, 1, 1, 0], device="cpu")
    predictions = torch.tensor([0, 1, 0, 0], device="cuda")

    try:
        e.calculate_f1_score(labels, predictions)
    except RuntimeError:
        pytest.fail("Device handling error!")


def test_get_model_unhappy_case():
    with pytest.raises(ValueError):
        e(model_name="unsupported_model", weights_path=None)


def test_get_model_happy_case():
    evaluator = e(model_name="resnet18", weights_path=None)
    got = evaluator.model

    assert isinstance(got, nn.Module)


def test_evaluator_passed_model_happy_case(binary_resnet18_fixture):
    evaluator = e(model=binary_resnet18_fixture)
    got = evaluator.model

    assert isinstance(got, nn.Module)
    assert got == binary_resnet18_fixture


def test_load_state_dict_happy_case(binary_resnet18_fixture):
    pretrained_model = binary_resnet18_fixture

    with tempfile.NamedTemporaryFile() as temp_file:
        torch.save(pretrained_model.state_dict(), temp_file.name)

        evaluator = e(model_name="resnet18", weights_path=temp_file.name)

        got = evaluator.model
        assert isinstance(got, nn.Module)

        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            assert torch.allclose(got.state_dict()[key], pretrained_state_dict[key], atol=1e-5)
