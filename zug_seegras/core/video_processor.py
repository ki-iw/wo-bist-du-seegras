from pathlib import Path

import cv2
import numpy as np
import torch


class VideoProcessor:
    def __init__(self, video_file: str, frame_ids: list[int], output_dir: str):
        self.video_file = Path(video_file)
        self.frame_ids = sorted(frame_ids)

        video_name = self.video_file.stem
        self.output_path = self._get_output_path(output_dir, video_name)

    @staticmethod
    def _get_output_path(output_dir: str, video_name: str) -> str:
        output_path = Path(output_dir) / video_name
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _get_total_frames(self) -> int:
        cap = cv2.VideoCapture(str(self.video_file))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {str(self.video_file)}")  # noqa: TRY003, RUF010
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def _get_frame_path(self, frame_id: int) -> Path:
        return self.output_path / f"frame_{frame_id:05d}.jpg"

    def _is_frame_saved(self, frame_id: int) -> bool:
        frame_path = self._get_frame_path(frame_id)
        return frame_path.exists()

    def _save_frame(self, frame_id: int, frame: np.ndarray) -> None:
        frame_path = self._get_frame_path(frame_id)
        cv2.imwrite(str(frame_path), frame)

    def _extract_frame(self, frame_id: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(self.video_file))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {str(self.video_file)}")  # noqa: TRY003, RUF010

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Failed to read frame {frame_id} from video.")  # noqa: TRY003
        return frame

    def extract_and_save_frames(self) -> None:
        total_frames = self._get_total_frames()

        for frame_id in self.frame_ids:
            if frame_id >= total_frames:
                raise ValueError(f"Frame ID {frame_id} is out of range for video with {total_frames} frames.")  # noqa: TRY003

            if self._is_frame_saved(frame_id):
                print(f"Frame {frame_id} already exists, skipping extraction.")
                continue

            frame = self._extract_frame(frame_id)
            self._save_frame(frame_id, frame)

    def load_frame_as_tensor(self, frame_id: int) -> torch.Tensor:
        frame_path = self._get_frame_path(frame_id)

        if not frame_path.exists():
            raise FileNotFoundError(f"Frame file {frame_path} does not exist.")  # noqa: TRY003

        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to load frame {frame_path}.")  # noqa: TRY003

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
        return frame_tensor

    def get_frames_for_dataloader(self) -> torch.Tensor:
        frame_list = [self.load_frame_as_tensor(frame_id) for frame_id in self.frame_ids]
        return torch.stack(frame_list)
