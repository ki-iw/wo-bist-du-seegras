from pathlib import Path

import cv2
import torch


class VideoProcessor:
    def __init__(self, video_file: str, frame_ids: list[int], output_dir: str):
        self.video_file = video_file
        self.frame_ids = sorted(frame_ids)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_save_frames(self) -> None:
        cap = cv2.VideoCapture(self.video_file)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")  # noqa: TRY003

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_id in self.frame_ids:
            if frame_id >= total_frames:
                raise ValueError(f"Frame ID {frame_id} is out of range for video with {total_frames} frames.")  # noqa: TRY003

            frame_path = self.output_dir / f"frame_{frame_id:05d}.jpg"

            if frame_path.exists():
                print(f"Frame {frame_id} already exists, skipping extraction.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_id} from video.")  # noqa: TRY003

            cv2.imwrite(str(frame_path), frame)

        cap.release()

    def get_frames_for_dataloader(self) -> torch.Tensor:
        frame_list = []

        for frame_id in self.frame_ids:
            frame_path = self.output_dir / f"frame_{frame_id:05d}.jpg"

            if not frame_path.exists():
                raise FileNotFoundError(f"Frame file {frame_path} does not exist.")  # noqa: TRY003

            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Failed to load frame {frame_path}.")  # noqa: TRY003

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
            frame_list.append(frame_tensor)

        frames_tensor = torch.stack(frame_list)

        return frames_tensor
