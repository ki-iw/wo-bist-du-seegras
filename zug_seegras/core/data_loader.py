from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose


def create_dataloaders(
    dataset_class,
    video_file: str,
    label_json_path: str,
    output_frames_dir: str,
    transform: Optional[Compose] = None,  # noqa: UP007
    batch_size: int = 4,
    train_test_ratio: float = 0.8,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset_class(
        video_file=video_file,
        label_dir=label_json_path,
        output_dir=output_frames_dir,
        transform=transform,
    )

    train_size = int(train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
