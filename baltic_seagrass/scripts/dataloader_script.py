from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from baltic_seagrass.core.datasets.seegras import SeegrasDataset
from baltic_seagrass.logger import getLogger

log = getLogger(__name__)


def main(input_path: str = "data/Seegras_v1"):
    data_path = Path(input_path)
    video_file = data_path / "input_video" / "trimmed_testvideo.mov"
    label_json_path = data_path / "input_label" / "default.json"
    output_frames_dir = data_path / "output"

    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    dataset = SeegrasDataset(
        video_file=str(video_file),
        label_dir=str(label_json_path),
        output_dir=str(output_frames_dir),
        transform=transforms,
    )

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in data_loader:
        log.info(f"Images shape: {images.shape}")
        log.info(f"Labels: {labels}")


if __name__ == "__main__":
    main()
