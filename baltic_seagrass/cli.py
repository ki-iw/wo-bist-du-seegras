import typer

from baltic_seagrass import getLogger

app = typer.Typer()
log = getLogger(__name__)


@app.command()
def train_quickstart():
    from baltic_seagrass.scripts.training_script import main

    main("resnet18")


@app.command()
def bag_of_seagrass_example():
    from baltic_seagrass.scripts.inference_patched_script import main

    main(save_path="data/patched", max_images=10)


def main():
    app()
