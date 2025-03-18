from typing import Annotated

import typer

from baltic_seagrass import getLogger

app = typer.Typer()
log = getLogger(__name__)


@app.command()
def train_quickstart():
    from baltic_seagrass.scripts.training_script import main

    main("resnet18")


@app.command()
def bag_of_seagrass_example(save_path="data/patched", max_images=4):
    from baltic_seagrass.scripts.inference_patched_script import main

    main(save_path=save_path, max_images=max_images)


@app.command()
def inference_on_video_example(
    video_file="/mnt/data/ZUG-Seegras/videos/DJI_20240923162615_0002_D_compressed50_14to16.MP4",
    save_path="data/inference_results",
    weights_path="data/model_checkpoints/resnet18/seagrass/resnet18_best-checkpoint.pth",
    skip: Annotated[int, typer.Option(help="Number of frames to skip between inference", show_default=True)] = 100,
):
    from baltic_seagrass.scripts.classifier_inference import main

    main(video_file=video_file, save_path=save_path, weights_path=weights_path, skip=skip)


def main():
    app()
