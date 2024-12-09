from pathlib import Path

import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from zug_seegras.core.data_loader import create_dataloaders
from zug_seegras.core.datasets.seegras import SeegrasDataset
from zug_seegras.core.evaluator import Evaluator
from zug_seegras.logger import getLogger

log = getLogger(__name__)


def main():
    data_path = Path("data")
    video_file = data_path / "input_video" / "trimmed_testvideo.mov"
    label_json_path = data_path / "input_label" / "default.json"
    output_frames_dir = data_path / "output"

    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    _, test_loader = create_dataloaders(
        dataset_class=SeegrasDataset,
        video_file=str(video_file),
        label_json_path=str(label_json_path),
        output_frames_dir=str(output_frames_dir),
        transform=transforms,
        batch_size=4,
        train_test_ratio=0.8,
        shuffle=True,
    )

    evaluator = Evaluator(model_name="seaclips", device="cuda" if torch.cuda.is_available() else "cpu")
    accuracy, f1_score = evaluator.run_evaluation(dataloader=test_loader)

    log.info(f"Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    main()
