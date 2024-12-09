import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from groundingdino.config import GroundingDINO_SwinT_OGC
from groundingdino.util.inference import load_model, predict
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from zug_seegras import getLogger
from zug_seegras.utils import convert_msec

warnings.filterwarnings("ignore")  # Suppress warnings
log = getLogger(__name__)


class Sampler:
    def __init__(
        self,
        input_path: str,
        output_path: str = "data/output",
        prompt: str = "underwater plants . seaweed . plant",
        model_name: str = "gDINO",
        skip_frame: int = 15,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        start_at: tuple[int, int, int] = (0, 0, 0),
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.prompt = prompt
        self.skip_frame = skip_frame
        self.start_at = start_at
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.run_folder = self.get_folder()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()

        self.annotations = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        self.annotation_id = 1

    def load_model(self) -> any:
        if self.model_name == "gDINO":
            log.info("Loading groundingDINO...")
            weights_path = Path.home() / "tmp/weights/groundingdino_swint_ogc.pth"
            return load_model(GroundingDINO_SwinT_OGC.__file__, weights_path)
        else:
            raise ValueError(f"Model {self.model_name} not implemented!")  # noqa: TRY003

    def get_folder(self) -> str:
        timestamp = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
        run_folder = self.output_path / f"run_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=True)

        (run_folder / "frames").mkdir(parents=True, exist_ok=True)
        (run_folder / "annotated_frames").mkdir(parents=True, exist_ok=True)

        return run_folder

    def get_annotations(
        self, image: np.ndarray, boxes: list[list[float]], labels: list[str], image_id: str, file_name: Path
    ) -> None:
        height, width = image.shape[:2]
        self.annotations["images"].append({"id": image_id, "height": height, "width": width, "file_name": file_name})

        unique_labels = set(labels)
        for label in unique_labels:
            if not any(c["name"] == label for c in self.annotations["categories"]):
                self.annotations["categories"].append(
                    {
                        "id": len(self.annotations["categories"]) + 1,
                        "name": label,
                    }
                )

        for i, box in enumerate(boxes):
            x_center, y_center, width_norm, height_norm = box
            w1 = int((x_center - width_norm / 2) * width)
            h1 = int((y_center - height_norm / 2) * height)
            w2 = int((x_center + width_norm / 2) * width)
            h2 = int((y_center + height_norm / 2) * height)

            coco_box = [w1, h1, w2 - w1, h2 - h1]

            category_id = next(c["id"] for c in self.annotations["categories"] if c["name"] == labels[i])
            annotation = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_box,
                "area": (w2 - w1) * (h2 - h1),
            }
            self.annotations["annotations"].append(annotation)
            self.annotation_id += 1

    def save_annotations(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(self.annotations, f)

    def draw_bounding_box(
        self, frame: np.ndarray, scores: list[float], boxes: list[list[float]], labels: list[str]
    ) -> np.ndarray:
        frame_bb = np.copy(frame)
        height, width = frame_bb.shape[:2]
        valid_boxes = []

        for n, box in enumerate(boxes):
            x_center, y_center, width_norm, height_norm = box
            w1 = int((x_center - width_norm / 2) * width)
            h1 = int((y_center - height_norm / 2) * height)
            w2 = int((x_center + width_norm / 2) * width)
            h2 = int((y_center + height_norm / 2) * height)

            # Check if the bounding box is too large (we don't want frame-sized boxes)
            if (w2 - w1) / width < 0.95 and (h2 - h1) / height < 0.95:
                valid_boxes.append((w1, h1, w2, h2, labels[n], scores[n]))

        for w1, h1, w2, h2, label, score in valid_boxes:
            cv2.rectangle(frame_bb, (w1, h1), (w2, h2), (0, 255, 0), 2)
            cv2.putText(
                frame_bb, f"{label}: {score:.2f}", (w1 + 8, h1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        return frame_bb if valid_boxes else None

    def save_run_info(self, output_folder: Path, detections: int, processed_frames: int) -> None:
        info = {
            "run_info": {
                "run_id": output_folder.name,
                "prompt": self.prompt,
                "skip_frames": self.skip_frame,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "model_name": self.model_name,
                "input_path": str(self.input_path),
                "output_folder": str(output_folder),
                "device": self.device,
            },
            "evaluation": {"detections": detections, "processed_frames": processed_frames},
        }
        config_path = output_folder / "run_info.yml"
        with open(config_path, "w") as outfile:
            yaml.dump(info, outfile, default_flow_style=False)

    def get_prediction(self, frame: np.ndarray) -> tuple[list[float], list[float], list[str]]:
        processed_frame = self.preprocess_image(frame)

        if self.model_name == "gDINO":
            boxes, scores, labels = predict(
                self.model, processed_frame, self.prompt, self.box_threshold, self.text_threshold
            )

        else:
            raise ValueError(f"Model {self.model_name} not implemented!")  # noqa: TRY003
        return boxes, scores, labels

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pil = Image.fromarray(image).convert("RGB")
        return transform(image_pil).to(self.device)

    def process_images(self, image_paths: list[str]):
        log.info(f"Start Image Processing. Saving results to {self.run_folder}")
        n_images = len(image_paths)
        processed_frames = 0
        detected = 0

        for n, image_path in enumerate(image_paths):
            try:
                image = cv2.imread(str(image_path))

                boxes, scores, labels = self.get_prediction(image)
                result = self.draw_bounding_box(image, scores, boxes, labels)

                if result is not None:
                    output_image_path = (
                        self.run_folder / "annotated_frames" / (image_path.stem + "_bb" + image_path.suffix)
                    )
                    cv2.imwrite(str(output_image_path), result)

                    detected += 1

                log.info(f"Processed image {n+1}/{n_images}.")
                processed_frames += 1

            except Exception:
                log.exception(f"Error processing image {image_path.stem}.")

        log.info(f"Finished Image Processing. Total frames with detection: {detected}")

    def process_video(self, video_path: Path) -> None:
        log.info(f"Start Video Processing: {video_path.stem}{video_path.suffix}. Saving results to {self.run_folder}")

        video = cv2.VideoCapture(str(video_path))
        frame_rate = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Start video delayed
        start_msec = (self.start_at[0] * 3600 + self.start_at[1] * 60 + self.start_at[2]) * 1000
        video.set(cv2.CAP_PROP_POS_MSEC, start_msec)

        frames_to_skip = int(start_msec / 1000 * frame_rate)
        n_frames = int((total_frames - frames_to_skip) / self.skip_frame)
        frame_count = 0
        processed_frames = 0
        detected = 0

        with tqdm(total=n_frames, desc="Processing Video", unit="frame", leave=True) as pbar:
            while True:
                try:
                    status, frame = video.read()
                    if not status:
                        break

                    if frame_count % self.skip_frame == 0:
                        hours, minutes, seconds, milliseconds = convert_msec(video.get(cv2.CAP_PROP_POS_MSEC))
                        boxes, scores, labels = self.get_prediction(frame)

                        result = self.draw_bounding_box(frame, scores, boxes, labels)
                        if result is not None:
                            output_frame_path = f"{self.run_folder}/frames/frame_{hours}h:{minutes}min:{seconds}sec:{milliseconds}msec.jpg"
                            output_bb_path = f"{self.run_folder}/annotated_frames/frame_bb_{hours}h:{minutes}min:{seconds}sec:{milliseconds}msec.jpg"
                            cv2.imwrite(output_bb_path, result)

                            self.get_annotations(frame, boxes, labels, processed_frames + 1, output_frame_path)
                            detected += 1

                        processed_frames += 1
                        pbar.set_description(f"Detections: {detected}")
                        pbar.update(1)

                    frame_count += 1

                except Exception as e:
                    log.exception(f"Exception: {e}")  # noqa: TRY401
                    break

        self.save_annotations(self.run_folder / "annotations.json")
        self.save_run_info(self.run_folder, detected, processed_frames)
        log.info(f"Finished processing {video_path.stem}. Total frames with detection: {detected}")

    def process_file(self) -> None:
        """Process input file and calls video/image processor."""
        image_paths = []

        try:
            # Process files
            if self.input_path.is_file():
                if self.input_path.suffix.lower() in [".mp4", ".mov"]:  # Process video file
                    self.process_video(self.input_path)
                    return
                elif self.input_path.suffix.lower() in [".jpg", "jpeg", ".png"]:  # Process image file
                    image_paths.append(self.input_path)
                    return
                else:
                    raise ValueError(f"Invalid file type: {self.input_path.suffix}")  # noqa: TRY003, TRY301
            # Process image folder
            if self.input_path.is_dir():
                image_paths.extend(self.input_path.glob("*.jpg"))
                image_paths.extend(self.input_path.glob("*.jpeg"))
                image_paths.extend(self.input_path.glob("*.png"))
            else:
                raise ValueError(f"Invalid input path: {self.input_path}")  # noqa: TRY003, TRY301

        except Exception as e:
            log.exception(e)  # noqa: TRY401
            raise

        self.process_images(image_paths)


if __name__ == "__main__":
    data_path = "/mnt/data/ZUG-Seegras/"
    input_path = "input/videos/Trident-Oct-01-124120-HQ.mp4"
    output_path = "/mnt/data/ZUG-Seegras/output"
    sampler = Sampler(os.path.join(data_path, input_path), output_path=output_path, start_at=(0, 0, 0))
    sampler.process_file()
