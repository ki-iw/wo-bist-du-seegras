import json
from pathlib import Path

import pytest

from zug_seegras.core.trainer import Trainer


@pytest.mark.parametrize("valid_json", [({"key": "value"}), ([{"key": "value"}, {"key": "value"}])])
def test_load_json_happy_case(tmp_path: Path, valid_json):
    valid_file = tmp_path / "valid.json"

    with open(valid_file, "w") as file:
        json.dump(valid_json, file)

    loaded_data = Trainer.load_config(str(valid_file))
    assert loaded_data == valid_json
