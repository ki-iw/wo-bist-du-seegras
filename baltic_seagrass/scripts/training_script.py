import warnings

from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from baltic_seagrass import logger
from baltic_seagrass.core.config_loader import get_model_config
from baltic_seagrass.core.data_loader import create_dataloaders
from baltic_seagrass.core.datasets.seagrass import SeagrassDataset
from baltic_seagrass.core.evaluator import Evaluator
from baltic_seagrass.core.fiftyone_logger import FiftyOneLogger
from baltic_seagrass.core.trainer import Trainer

warnings.filterwarnings("ignore", category=UserWarning, message="The reduce argument of torch.scatter")


def main(model_name: str):
    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    model_config = get_model_config(model_name)
    train_loader, test_loader = create_dataloaders(
        dataset_class=SeagrassDataset,
        transform=transforms,
        batch_size=model_config.training.batch_size,
        train_test_ratio=0.8,
        shuffle=model_config.training.shuffle,
    )

    trainer = Trainer(model_name=model_name, train_loader=train_loader, test_loader=test_loader, checkpoint_path=None)

    trainer.train()

    # latest model evaluation only to add to fiftyone!
    logger.info("Evaluating latest model!")
    trainer.evaluator.save_fiftyone = True
    trainer.evaluator.fiftyone_logger = FiftyOneLogger("Latest model")

    trainer.evaluator.run_evaluation(model=trainer.model, dataloader=trainer.test_loader)

    logger.info("Evaluating model with best F1 Score!")
    evaluator = Evaluator(
        model_name=model_name, checkpoint_path=trainer.best_checkpoint, n_classes=model_config.model.n_classes
    )
    evaluator.save_fiftyone = True
    evaluator.fiftyone_logger = FiftyOneLogger("Best model")
    evaluator.run_evaluation(dataloader=trainer.test_loader)

    evaluator.fiftyone_logger.visualize()


if __name__ == "__main__":
    model_name = "resnet18"
    main(model_name)
