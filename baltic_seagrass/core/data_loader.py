from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from baltic_seagrass import config


def split_dataset(dataset, train_test_ratio: float):
    train_size = int(train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def create_dataloaders(
    dataset_class,
    transform: Optional[Compose] = None,  # noqa: UP007
    batch_size: int = 4,
    train_test_ratio: float = 0.8,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset_class(
        video_files=config.dataset.video_files,
        annotations_dir=config.dataset.annotations_dir,  # directory containing json files named after the video files
        frames_dir=config.dataset.frames_dir,
        transform=transform,
    )

    train_dataset, test_dataset = split_dataset(dataset, train_test_ratio)

    train_loader = DataLoader(train_dataset, batch_size, shuffle)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader
