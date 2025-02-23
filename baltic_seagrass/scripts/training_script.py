import warnings

from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from baltic_seagrass.core.config_loader import get_model_config
from baltic_seagrass.core.data_loader import create_dataloaders
from baltic_seagrass.core.datasets.seegras import SeegrasDataset
from baltic_seagrass.core.evaluator import Evaluator
from baltic_seagrass.core.fiftyone_logger import FiftyOneLogger
from baltic_seagrass.core.trainer import Trainer

# from baltic_seagrass import config
warnings.filterwarnings("ignore", category=UserWarning, message="The reduce argument of torch.scatter")


def main(model_name: str):
    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    model_config = get_model_config(model_name)
    train_loader, test_loader = create_dataloaders(
        dataset_class=SeegrasDataset,
        transform=transforms,
        batch_size=model_config.training.batch_size,
        train_test_ratio=0.8,
        shuffle=True,
    )

    trainer = Trainer(model_name=model_name, train_loader=train_loader, test_loader=test_loader, checkpoint_path=None)

    trainer.train()

    # final evaluation with latest model
    trainer.evaluator.save_fiftyone = True
    trainer.evaluator.fiftyone_logger = FiftyOneLogger("Latest model")

    trainer.evaluator.run_evaluation(model=trainer.model, dataloader=trainer.test_loader)

    # final evaluation with best model
    evaluator = Evaluator(
        model_name=model_name, checkpoint_path=trainer.best_checkpoint, n_classes=trainer.model.n_classes
    )
    evaluator.save_fiftyone = True
    evaluator.fiftyone_logger = FiftyOneLogger("Best model")
    evaluator.run_evaluation(model=trainer.model, dataloader=trainer.test_loader)

    evaluator.fiftyone_logger.visualize()


if __name__ == "__main__":
    model_name = "resnet18"
    main(model_name)
