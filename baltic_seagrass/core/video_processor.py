from pathlib import Path

import cv2
import numpy as np
import torch

from zug_seegras import logger


class VideoProcessor:
    def set_output_path(self, output_dir: str, video_name: str) -> None:
        self.output_path = Path(output_dir) / video_name
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _get_frame_path(self, frame_id: int) -> Path:
        return self.output_path / f"frame_{frame_id:05d}.jpg"

    def _is_frame_saved(self, frame_id: int) -> bool:
        frame_path = self._get_frame_path(frame_id)
        return frame_path.exists()

    def _save_frame(self, frame_id: int, frame: np.ndarray) -> None:
        frame_path = self._get_frame_path(frame_id)
        cv2.imwrite(str(frame_path), frame)
        return frame_path

    def extract_and_save_frames(self, video_file: str, frame_ids: list[int]) -> None:
        frame_ids = sorted(frame_ids)
        cap = cv2.VideoCapture(Path(video_file))

        frame_paths = []
        for frame_id in frame_ids:
            frame_paths.append(self._get_frame_path(frame_id))
            if self._is_frame_saved(frame_id):
                logger.debug(f"Frame {frame_id} already exists, skipping extraction.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"Failed to read frame {frame_id} from video.")
                continue

            self._save_frame(frame_id, frame)

        cap.release()

        return frame_paths

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
