import pytest
import torch

from tests.utils import create_data_loader
from zug_seegras.core.evaluator import Evaluator as e


@pytest.fixture
def dummy_train_loader():
    return create_data_loader(batch_size=4, num_samples=10, image_size=(3, 512, 512))


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
        e(model_name="unsupported_model")


@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("model_name", ["seafeats", "seaclips", "resnet18", "seabag_ensemble", "grounding_dino"])
def test_run_evaluation_happy_case_integration(model_name, dummy_train_loader, n_classes):
    if n_classes == 4 and model_name not in ["seafeats", "seaclips", "seabag_ensemble"]:
        pytest.skip()

    evaluator = e(model_name=model_name, n_classes=n_classes)

    accuracy, f1_score = evaluator.run_evaluation(dataloader=dummy_train_loader)

    assert isinstance(accuracy, float)
    assert isinstance(f1_score, float)

    assert 0 <= accuracy <= 1
    assert 0 <= f1_score <= 1
