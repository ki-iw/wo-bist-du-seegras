from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from baltic_seagrass import logger
from baltic_seagrass.core.config_loader import get_model_config
from baltic_seagrass.core.data_loader import create_dataloaders
from baltic_seagrass.core.datasets.seagrass import SeagrassDataset
from baltic_seagrass.core.evaluator import Evaluator
from baltic_seagrass.core.fiftyone_logger import FiftyOneLogger
from baltic_seagrass.logger import getLogger

log = getLogger(__name__)


def main(model_name: str, checkpoint: str):
    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    model_config = get_model_config(model_name)
    _, test_loader = create_dataloaders(
        dataset_class=SeagrassDataset,
        transform=transforms,
        batch_size=model_config.evaluation.batch_size,
        train_test_ratio=0,  # creates test_loader with all data
        shuffle=model_config.evaluation.shuffle,
    )

    logger.info("Evaluating model of checkpoint %s!", checkpoint)
    evaluator = Evaluator(model_name=model_name, checkpoint_path=checkpoint, n_classes=model_config.model.n_classes)
    evaluator.save_fiftyone = True
    evaluator.fiftyone_logger = FiftyOneLogger("Results of checkpoint")
    evaluator.run_evaluation(dataloader=test_loader)

    evaluator.fiftyone_logger.visualize()


if __name__ == "__main__":
    model_name = "resnet18"
    checkpoint = "data/model_checkpoints/resnet18/seagrass/resnet18_best-checkpoint.pth"
    main(model_name, checkpoint)
