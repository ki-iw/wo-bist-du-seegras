from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import create_dataloaders
from zug_seegras.core.datasets.seegras import SeegrasDataset
from zug_seegras.core.evaluator import Evaluator
from zug_seegras.core.fiftyone_logger import FiftyOneLogger
from zug_seegras.core.trainer import Trainer


def main(model_name: str):
    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    train_loader, test_loader = create_dataloaders(
        dataset_class=SeegrasDataset,
        transform=transforms,
        batch_size=4,
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
