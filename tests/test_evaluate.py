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


@pytest.mark.parametrize(
    "labels, predictions",
    [
        (torch.tensor([0, 1]), torch.tensor([0])),
    ],
)
def test_calculate_accuracy_unhappy_case(labels, predictions):
    with pytest.raises(ValueError):
        e.calculate_accuracy(labels, predictions)
