# baltic-seagrass

A project to train, evaluate and run vision models to detect seagrass from images and video files.

## Setup

To get started, let's create a Conda environment, and install dependencies into this environment.

1. Create a new Conda environment:
    ```bash
    $ conda env create --file environment.yml
    ```
2. Activate the new environment!
    ```bash
    $ conda activate baltic-seagrass
    ```
3. Install the project dependencies into the newly created Conda environment.
    ```bash
    $ make install
    ```
4. Quick check whether everything has been installed as expected. Following will trigger the unit test execution:
    ```bash
    $ make test
    ```

## Quickstart
### Training
Run:
```
baltic_seagrass train-quickstart
```
This will start a quickstart of a training run with 20 epochs as defined in the [corresponding resnet18 config file](./baltic_seagrass/config/resnet18.yml) on the dataset you provide in the `dataset` fields of the [base.yml](./baltic_seagrass/config/base.yml). At the End of the run a fiftyone server is started highlighting the results of the evaluation on the testset. Checkpoints are written to `data/model_checkpoints` directory.

### Inference
### Bag of Seagrass Inference
Run:
```
baltic_seagrass bag-of-seagrass-example
```
This will create a `data/patched` directory and run the bag of seagrass inference on 4 frames loaded from the videos referenced in [base.yml](./baltic_seagrass/config/base.yml). After the execution, a fiftyone server is started where you can inspect the results in your browser.

### Behaviour

### Configuration

## Repository Structure

The repository is organized into the following key directories and scripts under the `baltic_seagrass/` folder:

### `baltic_seagrass/`
This is the main directory containing the productive code and key scripts for the project.

- **`core/`**: Contains the core logic and essential code for the project.
    - **`models/`**: Contains scripts for each model class. Currently, three models are implemented:
        - **`BinaryResnet18`**: A basic implementation of a PyTorch ResNet18 model with a binary output layer, trained using BinaryCrossEntropy loss.
        - **`BagOfSeagrass`**: Includes three separate models: SeaCLIP, SeaFeats, and an ensemble of both. These models can output a four-class result (background and three types of seagrass) or a binary classification.
        - **`GroundingDINO`**: Implementation of GroundingDINO, adjusted to the expected output format of the project. The bounding box output is post-processed to predict binary classification. This model is not finetunable.
    - **`datasets/`**: Contains scripts for building a PyTorch `Dataset` object from images and labels (processed using `datumaru_processor.py` and `video_processor.py`).
    - **`config_loader.py`**: Loads and processes the YAML configuration files.
    - **`data_loader.py`**: Constructs a PyTorch `DataLoader` object given a `Dataset` object (with training and test splits).
    - **`evaluator.py`**: Evaluates a model on a test `DataLoader` and computes metrics such as accuracy and F1 score. It can be used during training every `n` epochs or as a standalone script.
    - **`model_factory.py`**: Creates instances of the implemented models, and supports saving/loading model checkpoints (for resuming training).
    - **`trainer.py`**: Manages the training of models based on the YAML configuration file, supports model checkpointing, and provides evaluation at specified intervals.
    - **`video_processor.py`**: Processes video files by extracting relevant frames (based on labels) and applying basic transformations.

- **`scripts/`**: Contains scripts for interacting with the core functionality of the repository.
    - **`dataloader_script.py`**: Preprocesses a video and its associated label JSON file, extracting frames and creating the appropriate folder structure for data.
    - **`evaluation_script.py`**: Evaluates a trained model on a dataset (created by `dataloader_script.py`), and generates evaluation metrics.
    - **`training_script.py`**: Trains a model on a given dataset folder and evaluates it periodically.

- **`config/`**: Contains configuration files in YAML format for training, evaluation, and dataset selection.
    - **`base.yml`**: Base configuration shared by all models, containing parameters for evaluation and dataset settings.
    - **`[model_name].yml`**: Model-specific YAML files that handle the training parameters for each model.

### Data Processing Workflow
The data processing pipeline is as follows:
1. **Video and Label Files**: Given a video file and a label JSON file (currently in Datumaru format), we construct a dataset by processing the labels (`datumaru_processor.py`) and extracting relevant frames from the video (`video_processor.py`).
2. **Dataset Organization**: Each video is treated as a separate dataset (though in the future, there may be an option to combine frames from multiple videos into a single dataset).
3. **Dataset and DataLoader Creation**: The `datasets/seagrass.py` and `video_processor.py` scripts are used to create the `Dataset` and `DataLoader` objects that are compatible with PyTorch for training, inference, and evaluation.


## Running

The project includes scripts to perform key tasks, which are located in the `scripts` subfolder:

- **Training**: Use `training_script.py` to handle data loading, model training, and checkpointing.
- **Evaluation**: Use `evaluation_script.py` to evaluate trained models and generate metrics such as accuracy and F1 score.
- **Data Loading**: Use `dataloader_script.py` to preprocess and load data.

These scripts ensure the project is modular, reproducible, and easy to extend.


## Development
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

## FiftyOne Integration

When running the evaluator with the argument `save_fiftyone=True`, the data is saved to a FiftyOne dataset. To view the dataset in the FiftyOne web UI, execute the `fiftyone_logger.py` script.

If you are working on the KI-IW remote machine, you may need to manually forward port `5154` to your local machine. Once the port is forwarded, you can access the FiftyOne web interface by navigating to [http://localhost:5154/](http://localhost:5154/) in your browser.

> Known Issue: