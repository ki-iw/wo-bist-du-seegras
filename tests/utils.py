import pytest
import torch
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    def __init__(self, x_data, y_data, paths):
        self.x_data = x_data
        self.y_data = y_data
        self.paths = paths

        assert len(self.x_data) == len(self.y_data) == len(self.paths)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx], self.paths[idx]


def create_data_loader(batch_size: int, num_samples: int, image_size: tuple):
    x_data = torch.randn(num_samples, *image_size)
    y_data = torch.randint(0, 2, (num_samples,))

    paths = [f"/path/to/image_{i}.jpg" for i in range(num_samples)]

    dataset = DummyDataset(x_data, y_data, paths)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def dummy_train_loader():
    return create_data_loader(batch_size=4, num_samples=10, image_size=(3, 512, 512))
