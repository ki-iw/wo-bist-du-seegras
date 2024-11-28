import pytest
import torch

from zug_seegras.core.evaluation import Evaluator as e


@pytest.mark.parametrize(
    "labels, predictions, want",
    [
        (torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]), 1.0),
        (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 0, 1]), 0.5),
        (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 1, 0]), 0.0),
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
