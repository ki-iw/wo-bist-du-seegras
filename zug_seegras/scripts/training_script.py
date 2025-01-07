from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import create_dataloaders
from zug_seegras.core.datasets.seegras import SeegrasDataset
from zug_seegras.core.trainer import Trainer


def main(model_name: str, data_path: str = "data/Seegras_v1"):
    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    train_loader, test_loader = create_dataloaders(
        dataset_class=SeegrasDataset,
        dataset_dir=data_path,
        transform=transforms,
        batch_size=4,
        train_test_ratio=0.8,
        shuffle=True,
    )

    trainer = Trainer(model_name=model_name, train_loader=train_loader, test_loader=test_loader, checkpoint_path=None)

    trainer.train()


if __name__ == "__main__":
    model_name = "resnet18"
    main(model_name)
