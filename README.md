# ZUG-seegras

A project to train, evaluate and run vision models to detect seagrass from images and video files.

## Setup

To get started, let's create a Conda environment, and install dependencies into this environment.

1. Change into the newly created project folder `ZUG-seegras`, and initialize the Git repo
    ```bash
    $ git init -b main
    ```
1. Create a new Conda environment:
    ```bash
    $ conda env create --file environment.yml
    ```
1. Activate the new environment!
    ```bash
    $ conda activate bom-ZUG-seegras
    ```
1. Install the project dependencies into the newly created Conda environment.
    ```bash
    $ make install
    ```
1. Lastly, run the project
    ```bash
    $ python -m zug_seegras
    ```

## Running

The project includes scripts to perform key tasks, which are located in the `scripts` subfolder:

- **Training**: Use `training_script.py` to handle data loading, model training, and checkpointing.
- **Evaluation**: Use `evaluation_script.py` to evaluate trained models and generate metrics such as accuracy and F1 score.
- **Data Loading**: Use `dataloader_script.py` to preprocess and load data.

These scripts ensure the project is modular, reproducible, and easy to extend.


## Developement
Some tasks need to be done repeatedly.

### Adding dependencies
Use `poetry` to add new dependencies to the project:
```bash
$ poetry add [package-name]
```

### Running tests
Run all unit and integration tests, and print a coverage report:
```bash
$ make test
```