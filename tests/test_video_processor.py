from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from baltic_seagrass.core.video_processor import VideoProcessor


def create_dummy_video(video_path: str, num_frames: int = 10, width: int = 640, height: int = 480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20 % 255, i * 40 % 255, i * 60 % 255), dtype=np.uint8)
        out.write(frame)

    out.release()


@pytest.fixture
def video_processor_fixture(tmp_path):
    video_path = tmp_path / "dummy_video.mp4"
    output_dir = tmp_path / "output_frames"

    create_dummy_video(str(video_path), num_frames=10)

    video_processor = VideoProcessor()
    video_processor.set_output_path(output_dir, Path(video_path).stem)

    yield video_processor


def test_get_frame_path_happy_case(tmp_path, video_processor_fixture):
    want = tmp_path / "output_frames" / "dummy_video" / "frame_00004.jpg"
    got = video_processor_fixture._get_frame_path(4)

    assert got == want


@patch.object(VideoProcessor, "_is_frame_saved", return_value=True)
def test_extract_and_save_frames_happy_case_frame_exists(video_processor_fixture, tmp_path):
    video_path = tmp_path / "dummy_video.mp4"
    frame_ids = [0, 2, 4]

    video_processor_fixture.extract_and_save_frames(video_path, frame_ids)


def test_load_frame_as_tensor_file_unhappy_case_not_found(video_processor_fixture):
    with patch.object(
        video_processor_fixture, "_get_frame_path", return_value=Path("non_existent_frame.jpg")
    ), pytest.raises(FileNotFoundError):
        video_processor_fixture.load_frame_as_tensor(0)


def test_load_frame_as_tensor_file_unhappy_case_not_exist(video_processor_fixture):
    with patch.object(Path, "exists", return_value=True), patch("cv2.imread", return_value=None), pytest.raises(
        ValueError
    ):
        video_processor_fixture.load_frame_as_tensor(0)
