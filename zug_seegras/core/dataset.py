from pathlib import Path
from typing import Optional

import cv2
import torch
from torch.utils.data import Dataset

from zug_seegras.core.datumaru_processor import DatumaroProcessor
from zug_seegras.core.video_processor import VideoProcessor


class SeegrasDataset(Dataset):
    def __init__(
        self,
        video_file: str,
        label_dir: str,
        output_dir: str,
        processor=DatumaroProcessor,
        transform: Optional[any] = None,  # noqa: UP007
    ) -> None:
        self.video_file = video_file
        self.output_dir = Path(output_dir)
        self.transform = transform

        datumaro_processor = processor(label_dir)
        self.frame_ids, self.labels = datumaro_processor.get_frame_labels()

        self.video_processor = VideoProcessor(
            video_file=self.video_file, frame_ids=self.frame_ids, output_dir=self.output_dir
        )
        self.video_processor.extract_and_save_frames()

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        label = self.labels[idx]

        frame_path = self.output_dir / Path(self.video_file).stem / f"frame_{frame_id:05d}.jpg"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame {frame_id} could not be found at {frame_path}.")  # noqa: TRY003

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Failed to load frame {frame_path}.")  # noqa: TRY003

        if self.transform:
            image = self.transform(image)

        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image_tensor, label


if __name__ == "__main__":
    data_path = Path("data")

    video_file = data_path / "input_video" / "trimmed_testvideo.mov"
    label_json_path = data_path / "input_label" / "default.json"
    output_frames_dir = data_path / "output"

    dataset = SeegrasDataset(
        video_file=str(video_file),
        label_dir=str(label_json_path),
        output_dir=str(output_frames_dir),
    )
