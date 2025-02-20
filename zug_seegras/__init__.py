import os
from importlib import metadata
from importlib.metadata import version

from dotenv import load_dotenv
from dotmap import DotMap

from zug_seegras.core.config_loader import load_config
from zug_seegras.logger import getLogger

load_dotenv()

try:
    __version__ = version("zug_seegras")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

config = DotMap(load_config("zug_seegras/config/base.yml"))
logger = getLogger(__name__)
logger.setLevel(config.log_level)

os.environ["FIFTYONE_DATABASE_DIR"] = config.fiftyone.dataset_dir

__all__ = [
    "getLogger",
]
