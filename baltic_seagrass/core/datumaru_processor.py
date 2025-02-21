import json

from baltic_seagrass.logger import getLogger

log = getLogger(__name__)


class DatumaroProcessor:
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

    def get_label_name(self, label_id: int) -> str:
        return self.label_mapping[label_id]

    def get_frame_labels(self, json_file: str) -> tuple[list[int], list[int], list[int]]:
        json_data = self.load_json(json_file)
        self.label_mapping = json_data["categories"]["label"]["labels"]
        items = json_data.get("items", [])
        frame_ids = []
        labels = []
        invalid_frames = []

        for item in items:
            item_id = int(item.get("attr").get("frame"))
            annotations = item.get("annotations", [])

            if not annotations:
                continue

            if len(annotations) > 1:
                log.info(f"Number of labels in frame {item_id}: {len(annotations)}")
                invalid_frames.append(item_id)
                continue

            frame_ids.append(item_id)
            label_id = annotations[0].get("label_id")
            labels.append(label_id)

        return frame_ids, labels, invalid_frames


if __name__ == "__main__":
    datumaro_processor = DatumaroProcessor()
    frame_ids, labels, invalid_frames = datumaro_processor.get_frame_labels(
        "tmp/annotations/DJI_20240923162615_0002_D_compressed50_10to12.json"
    )
