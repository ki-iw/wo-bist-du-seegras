from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from zug_seegras import config


def get_file_paths(dataset_dir: str):
    dataset_path = Path(dataset_dir)

    video_file = next(dataset_path.glob("input_video/*.MP4"), None)
    label_json_path = next(dataset_path.glob("input_label/*.json"), None)
    output_frames_dir = dataset_path / "output"

    if video_file is None or label_json_path is None:
        raise FileNotFoundError(f"Video file or label file not found in {dataset_path}.")  # noqa: TRY003

    return video_file, label_json_path, output_frames_dir


def split_dataset(dataset, train_test_ratio: float):
    train_size = int(train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def get_dataloader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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

    train_loader = get_dataloader(train_dataset, batch_size, shuffle)
    test_loader = get_dataloader(test_dataset, batch_size, shuffle=False)

    return train_loader, test_loader
