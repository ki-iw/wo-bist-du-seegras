import os
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from baltic_seagrass.core.fiftyone_logger import FiftyOneLogger
from baltic_seagrass.core.model_factory import ModelFactory
from baltic_seagrass.logger import getLogger

log = getLogger(__name__)


def main(video_file, save_path, weights_path, max_images=10, skip=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    os.makedirs(save_path, exist_ok=True)

    model_factory = ModelFactory(device)
    classifier = model_factory.create_model(model_name="resnet18", n_classes=1)
    model_factory.load_checkpoint(classifier, weights_path)

    classifier.eval()

    fiftyone_logger = FiftyOneLogger(dataset_name="classifier_predictions")

    cap = cv2.VideoCapture(Path(video_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Playing video", unit="frame"):
            if (frame_idx % skip) != 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:  # If video ends unexpectedly, break the loop
                continue

            # TODO is this necessary?
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)

            frame = transforms(pil_frame).unsqueeze(0).to(device)

            outputs = classifier(frame)
            probs = torch.sigmoid(outputs).squeeze()
            predicted = (probs > 0.5).long()

            video_name = Path(video_file).stem
            filename = os.path.join(save_path, f"{video_name}_frame{frame_idx}.png")
            pil_frame.save(filename)

            fiftyone_logger.add_sample(filename, -1, predicted)

    cap.release()
    fiftyone_logger.visualize()


if __name__ == "__main__":
    main(
        video_file="/mnt/data/ZUG-Seegras/videos/DJI_20240923162615_0002_D_compressed50_14to16.MP4",
        save_path="data/inference_results",
        weights_path="data/model_checkpoints/resnet18/seagrass/resnet18_best-checkpoint.pth",
        max_images=4,
    )
