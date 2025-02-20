import pytest

from tests.utils import create_data_loader
from zug_seegras.core.evaluator import Evaluator as e


@pytest.fixture
def dummy_train_loader():
    return create_data_loader(batch_size=4, num_samples=10, image_size=(3, 512, 512))


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
