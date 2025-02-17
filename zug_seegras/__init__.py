from importlib import metadata
from importlib.metadata import version

from dotenv import load_dotenv
from dotmap import DotMap

from zug_seegras.core.config_loader import get_model_config
from zug_seegras.logger import getLogger

load_dotenv()

try:
    __version__ = version("zug_seegras")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

# TODO remove modelname arg and make it non hardcoded
config = DotMap(get_model_config("resnet18"))
logger = getLogger(__name__)
logger.setLevel(config.log_level)

__all__ = [
    "getLogger",
]
