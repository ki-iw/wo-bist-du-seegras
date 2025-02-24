import os
from pathlib import Path

import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# from baltic_seagrass import config
from baltic_seagrass.core.data_loader import create_dataloaders
from baltic_seagrass.core.datasets.seagrass import SeagrassDataset
from baltic_seagrass.core.fiftyone_logger import FiftyOneLogger
from baltic_seagrass.core.model_factory import ModelFactory
from baltic_seagrass.logger import getLogger
from baltic_seagrass.utils import cropper, denormalize, img_to_grid, single_image_preds

log = getLogger(__name__)


def main(save_path="data/Seegras_v1/patched_preds", max_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO put this in config
    transform = Compose(
        [Resize((2048, 2048)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    _, test_loader = create_dataloaders(
        dataset_class=SeagrassDataset,
        transform=transform,
        batch_size=1,
        train_test_ratio=0.8,
        shuffle=True,
    )

    model_factory = ModelFactory(device)
    seafeats = model_factory.create_model(model_name="seafeats", n_classes=4)
    seaclip = model_factory.create_model(model_name="seaclips", n_classes=4)
    seabag_ensemble = model_factory.create_model(model_name="seabag_ensemble", n_classes=4)

    seafeats.eval()
    seaclip.eval()
    seabag_ensemble.eval()

    fiftyone_logger = FiftyOneLogger(dataset_name="patched_preds")
    max_images = len(test_loader) * test_loader.batch_size if max_images < 0 else max_images
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i + 1 > max_images:
                break

            inputs, labels, paths = batch

            log.info(f"Processing batch {i} with {len(inputs)} images.")

            inputs, labels = inputs.to(device), labels.to(device)

            vis_image = inputs[0].cpu().permute(1, 2, 0).numpy()

            row, col = 5, 9
            grid, _, _ = img_to_grid(vis_image, row, col)

            all_patches = []
            for patch in grid:
                patch_crop = cropper(patch, int(vis_image.shape[1] / col), int(vis_image.shape[0] / row))
                all_patches.append(patch_crop.unsqueeze(0))
                all_patches_torch = torch.cat(all_patches, dim=0).to(device)

            outputs_list, cos_list, clip_list = [], [], []
            for patch_torch in all_patches_torch:
                outputs_cos = seafeats(patch_torch.unsqueeze(0))
                preds_cos = torch.argmax(outputs_cos, dim=1).int()

                outputs_clip = seaclip(patch_torch.unsqueeze(0))
                preds_clip = torch.argmax(outputs_clip, dim=1).int()

                outputs_ensemble = seabag_ensemble(patch_torch.unsqueeze(0))
                preds_torch = torch.argmax(outputs_ensemble, dim=1).int()

                outputs_list.append(preds_torch)
                cos_list.append(preds_cos)
                clip_list.append(preds_clip)

            outputs_torch = torch.cat(outputs_list, dim=0)
            outputs_torch_cos = torch.cat(cos_list, dim=0)
            outputs_torch_clip = torch.cat(clip_list, dim=0)

            path = Path(paths[0]).stem
            whole_image = denormalize(inputs[0])
            clip_name = os.path.join(save_path, f"{path}_clip.png")
            cos_name = os.path.join(save_path, f"{path}_cos.png")
            ensemble_name = os.path.join(save_path, f"{path}_ensemble.png")
            single_image_preds(outputs_torch_clip, whole_image, clip_name)
            single_image_preds(outputs_torch_cos, whole_image, cos_name)
            single_image_preds(outputs_torch, whole_image, ensemble_name)

            fiftyone_logger.add_image(clip_name, {"predicted_by": "seaclip"})
            fiftyone_logger.add_image(cos_name, {"predicted_by": "seafeats"})
            fiftyone_logger.add_image(ensemble_name, {"predicted_by": "ensemble"})

        fiftyone_logger.visualize()


if __name__ == "__main__":
    main(save_path="data/patched", max_images=-1)
