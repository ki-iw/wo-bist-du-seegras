import pytest
import torch

from zug_seegras.core.evaluation import Evaluator as e


@pytest.mark.parametrize(
    "labels, predictions, expected_accuracy",
    [
        (torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]), 1.0),
        (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 0, 1]), 0.5),
        (torch.tensor([0, 1, 0, 1]), torch.tensor([1, 0, 1, 0]), 0.0),
    ],
)
def test_calculate_accuracy_happy_case(labels, predictions, expected_accuracy):
    accuracy = e.calculate_accuracy(labels, predictions)
    assert accuracy == expected_accuracy
    assert isinstance(accuracy, float)


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
