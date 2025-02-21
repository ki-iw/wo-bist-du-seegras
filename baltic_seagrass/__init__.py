import os
from importlib import metadata
from importlib.metadata import version

from dotenv import load_dotenv
from dotmap import DotMap

from baltic_seagrass.core.config_loader import load_config
from baltic_seagrass.logger import getLogger

load_dotenv()

try:
    __version__ = version("baltic_seagrass")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

config = DotMap(load_config("baltic_seagrass/config/base.yml"))
logger = getLogger(__name__)
logger.setLevel(config.log_level)

os.environ["FIFTYONE_DATABASE_DIR"] = config.fiftyone.dataset_dir

__all__ = [
    "getLogger",
]
