from pathlib import Path
from typing import Optional

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from zug_seegras.core.datumaru_processor import DatumaroProcessor
from zug_seegras.core.video_processor import VideoProcessor
from zug_seegras.logger import getLogger

log = getLogger(__name__)


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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image = self.transform(image) if self.transform else pil_to_tensor(image).float() / 255.0
        return image, label
