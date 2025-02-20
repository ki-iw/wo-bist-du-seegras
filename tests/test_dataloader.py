from typing import Optional

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import zug_seegras.core.data_loader as d


class DummyDataset(Dataset):
    def __init__(
        self,
        video_file: Optional[str] = None,  # noqa: UP007
        label_dir: Optional[str] = None,  # noqa: UP007
        output_dir: Optional[str] = None,  # noqa: UP007
        num_samples=10,
        transform=None,
    ):
        self.data = []
        self.labels = []

        for i in range(num_samples):
            image = np.random.rand(64, 64, 3) * 255
            image = Image.fromarray(image.astype("uint8"))
            self.data.append(image)
            self.labels.append(i % 2)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


@pytest.fixture
def dataset_fixture():
    return DummyDataset


@pytest.fixture
def dataset_dir_fixture(tmp_path):
    dataset_dir = tmp_path / "fake_data_dir"
    dataset_dir.mkdir()

    (dataset_dir / "input_video").mkdir()
    (dataset_dir / "input_label").mkdir()
    (dataset_dir / "output").mkdir()

    (dataset_dir / "input_video" / "fake_video.mov").touch()
    (dataset_dir / "input_label" / "fake_labels.json").touch()

    return str(dataset_dir)


def test_split_dataset_happy_case(dataset_fixture):
    dataset = dataset_fixture()
    train_dataset, test_dataset = d.split_dataset(dataset, 0.8)

    assert len(train_dataset) == 8
    assert len(test_dataset) == 2


def test_get_dataloader_happy_case(dataset_fixture):
    dataset = dataset_fixture()
    train_loader = d.get_dataloader(dataset, batch_size=4, shuffle=True)

    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 4
