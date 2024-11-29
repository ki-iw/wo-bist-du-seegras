import json

import torch


class DatumaroProcessor:
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.json_data = self.load_json(json_file)

    @staticmethod
    def load_json(json_file: str) -> dict[str, any]:
        try:
            with open(json_file) as file:
                json_data = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError from e
        except json.JSONDecodeError as e:
            raise ValueError from e

        return json_data

    def convert_datumaru(self) -> tuple[list[int], torch.Tensor]:
        items = self.json_data.get("items", [])
        frame_ids = []
        labels = []

        last_labeled_index = -1
        for index, item in enumerate(items):
            annotations = item.get("annotations", [])
            if any(annotation.get("type") == "bbox" for annotation in annotations):
                last_labeled_index = index

        for index, item in enumerate(items):
            if index > last_labeled_index:
                break

            item_id = int(item.get("attr").get("frame"))
            annotations = item.get("annotations", [])

            if not annotations:
                frame_ids.append(item_id)
                labels.append(0)
                continue

            has_bbox = any(annotation.get("type") == "bbox" for annotation in annotations)

            if not has_bbox:
                continue

            frame_ids.append(item_id)
            labels.append(1)

        return frame_ids, torch.tensor(labels, dtype=torch.int)

    def get_frame_labels(self) -> tuple[list[int], torch.Tensor]:
        return self.convert_datumaru()


if __name__ == "__main__":
    processor = DatumaroProcessor("data/input_label/default.json")
    print(processor.get_frame_labels())
