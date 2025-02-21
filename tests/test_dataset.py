from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from baltic_seagrass.core.datasets.seegras import SeegrasDataset


@pytest.fixture
def mock_datumaro_processor():
    with patch("baltic_seagrass.core.datasets.seegras.DatumaroProcessor") as MockDatumaroProcessor:
        mock_instance = MockDatumaroProcessor.return_value
        mock_instance.get_frame_labels.return_value = ([1, 2], [0, 1])
        yield MockDatumaroProcessor


@pytest.fixture
def mock_dataset(tmp_path, mock_datumaro_processor):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    image_dir = output_dir / "testvideo"
    image_dir.mkdir(parents=True)
    dummy_image = Image.new("RGB", (1024, 1024))
    dummy_image.save(image_dir / "frame_00001.jpg")
    dummy_image.save(image_dir / "frame_00002.jpg")

    return SeegrasDataset(
        video_files=["testvideo.mov"],
        annotations_dir="mock_label_dir",
        frames_dir=output_dir,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "transform, want",
    [
        (None, (3, 1024, 1024)),
        (Compose([Resize((512, 512)), ToTensor()]), (3, 512, 512)),
    ],
)
def test_get_item_with_transform_happy_case(mock_dataset, batch_size, transform, want):
    mock_dataset.transform = transform

    dataloader = DataLoader(mock_dataset, batch_size=batch_size, shuffle=False)

    for batch_images, batch_labels, batch_paths in dataloader:
        assert batch_images.shape == (batch_size, *want)
        assert batch_labels.shape == (batch_size,)
        assert len(batch_paths) == batch_size
        assert isinstance(batch_paths, tuple) and all(isinstance(path, str) for path in batch_paths)
        assert isinstance(batch_images, torch.Tensor)
        assert isinstance(batch_labels, torch.Tensor)
