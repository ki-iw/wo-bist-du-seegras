import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from baltic_seagrass.core.data_loader import create_dataloaders
from baltic_seagrass.core.datasets.seagrass import SeagrassDataset
from baltic_seagrass.core.evaluator import Evaluator
from baltic_seagrass.logger import getLogger

log = getLogger(__name__)


def main(model_name: str, data_path: str = "data/Seegras_v1"):
    transforms = Compose(
        [Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    _, test_loader = create_dataloaders(
        dataset_class=SeagrassDataset,
        dataset_dir=data_path,
        transform=transforms,
        batch_size=4,
        train_test_ratio=0.8,
        shuffle=True,
    )

    evaluator = Evaluator(
        model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu", save_fiftyone=True
    )
    accuracy, f1_score = evaluator.run_evaluation(dataloader=test_loader)

    log.info(f"Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    model_name = "resnet18"
    main(model_name)
