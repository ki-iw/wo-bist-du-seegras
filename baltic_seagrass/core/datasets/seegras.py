from pathlib import Path
from typing import Optional

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from baltic_seagrass import logger
from baltic_seagrass.core.datumaru_processor import DatumaroProcessor
from baltic_seagrass.core.video_processor import VideoProcessor


# TODO change to seagrass
class SeegrasDataset(Dataset):
    def __init__(
        self,
        video_files: list[str],
        annotations_dir: str,  # directory containing json files named after the video files
        frames_dir: str,  # directory to save the extracted frames
        label_processor=DatumaroProcessor,
        transform: Optional[any] = None,  # noqa: UP007
    ) -> None:
        self.video_files = video_files
        self.transform = transform

        self.frame_paths = []
        self.labels = []

        label_processor = label_processor()
        self.video_processor = VideoProcessor()
        for video_file in self.video_files:
            self.video_processor.set_output_path(frames_dir, Path(video_file).stem)
            label_path = Path(annotations_dir) / Path(video_file).with_suffix(".json").name

            if not label_path.exists():
                logger.warning(f"Label file not found for {video_file} inside {annotations_dir}. Skipping ...")
                continue

            frame_ids, labels, _ = label_processor.get_frame_labels(label_path)

            self.labels.extend(labels)

            logger.info("Extracting and saving frames ...")
            frame_paths = self.video_processor.extract_and_save_frames(video_file, frame_ids)
            self.frame_paths.extend(frame_paths)

        logger.info(f"Total frames: {len(self.frame_paths)}")
        logger.info(f"Total labels: {len(self.labels)}")
        for label_id in set(self.labels):
            logger.info(
                f"Total number of label {label_processor.get_label_name(label_id)['name']}: {self.labels.count(label_id)}"
            )

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]

        frame_path = self.frame_paths[idx]
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame could not be found at {frame_path}.")  # noqa: TRY003

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"Failed to load frame {frame_path}.")  # noqa: TRY003

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image = self.transform(image) if self.transform else pil_to_tensor(image).float() / 255.0
        return image, label, str(frame_path)
