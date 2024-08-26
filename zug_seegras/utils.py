import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def convert_msec(msec):
    """Convert milliseconds to hours, minutes, seconds, and milliseconds."""
    milliseconds = int((msec % 1000) / 100)
    seconds = int(msec / 1000) % 60
    minutes = int(msec / (1000 * 60)) % 60
    hours = int(msec / (1000 * 60 * 60)) % 24
    return hours, minutes, seconds, milliseconds


def draw_bounding_box(
    frame: np.ndarray, scores: list[float], boxes: list[list[float]], labels: list[str]
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
        cv2.putText(frame_bb, f"{label}: {score:.2f}", (w1 + 8, h1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame_bb if valid_boxes else None


def preprocess_image(image: np.ndarray, device: str = "cuda") -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(image).convert("RGB")
    return transform(image_pil).to(device)
