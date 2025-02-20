import json
from pathlib import Path
from unittest.mock import patch

import pytest

from zug_seegras.core.datumaru_processor import DatumaroProcessor as D

sample_datumaro_json = {
    "categories": {
        "label": {
            "labels": [
                {"name": "seagrass", "parent": "", "attributes": []},
                {"name": "background", "parent": "", "attributes": []},
            ],
            "attributes": ["occluded"],
        },
        "points": {"items": []},
    },
    "items": [
        {"id": "frame_000001", "attr": {"frame": 1}, "annotations": []},
        {"id": "frame_000002", "attr": {"frame": 2}, "annotations": [{"type": "bbox", "label_id": 0}]},
        {"id": "frame_000003", "attr": {"frame": 3}, "annotations": [{"type": "bbox", "label_id": 1}]},
        {
            "id": "frame_000004",
            "attr": {"frame": 4},
            "annotations": [{"type": "bbox", "label_id": 0}, {"type": "bbox", "label_id": 1}],
        },
        {
            "id": "frame_000005",
            "attr": {"frame": 5},
            "annotations": [{"type": "bbox", "label_id": 0}, {"type": "bbox", "label_id": 0}],
        },
        {
            "id": "frame_000006",
            "attr": {"frame": 6},
            "annotations": [{"type": "bbox", "label_id": 1}, {"type": "bbox", "label_id": 1}],
        },
        {"id": "frame_000007", "attr": {"frame": 7}, "annotations": []},
        {"id": "frame_000008", "attr": {"frame": 8}, "annotations": []},
    ],
}


@pytest.mark.parametrize("valid_json", [({"key": "value"}), ([{"key": "value"}, {"key": "value"}])])
def test_load_json_happy_case(tmp_path: Path, valid_json):
    valid_file = tmp_path / "valid.json"

    with open(valid_file, "w") as file:
        json.dump(valid_json, file)

    loaded_data = D.load_json(str(valid_file))
    assert loaded_data == valid_json


def test_load_json_file_unhappy_case_not_found():
    with pytest.raises(FileNotFoundError):
        D.load_json("non_existent_file.json")


def test_load_json__unhappy_case_invalid_json(tmp_path: Path):
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as file:
        file.write("{invalid json}")

    with pytest.raises(ValueError):
        D.load_json(str(invalid_file))


@patch.object(D, "load_json", return_value=sample_datumaro_json)
def test_convert_datumaru_happy_case(mock_load_json):
    processor = D()
    frame_ids, labels, invalid_frames = processor.get_frame_labels(json_file="dummy_path.json")

    expected_frame_ids = [2, 3]
    expected_invalid_frames = [4, 5, 6]
    expected_labels = [0, 1]

    assert frame_ids == expected_frame_ids
    assert labels == expected_labels
    assert invalid_frames == expected_invalid_frames
