from importlib import metadata
from importlib.metadata import version

from dotenv import load_dotenv

from .core import BaseClass
from .logger import getLogger

load_dotenv()

try:
    __version__ = version("zug_seegras")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "BaseClass",
    "getLogger",
]
