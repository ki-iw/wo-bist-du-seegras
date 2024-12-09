from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import SeegrasDataset
from zug_seegras.core.trainer import Trainer

if __name__ == "__main__":
    data_path = Path("data")

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

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    config_path = "zug_seegras/config/config.yml"

    trainer = Trainer(config_path=config_path, train_loader=train_loader, test_loader=test_loader, checkpoint_path=None)

    trainer.train()
