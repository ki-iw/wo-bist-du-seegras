import argparse

from baltic_seagrass import __version__, getLogger

log = getLogger(__name__)

parser = argparse.ArgumentParser(
    description="baltic_seagrass",
    epilog=f"Version {__version__}",
)
parser.add_argument(
    "--log-level",
    dest="log_level",
    type=str,
    default="INFO",
    help="Log level. Use DEBUG to debug.",
)


def main() -> None:
    args = parser.parse_args()
    log.setLevel(args.log_level)
    log.info(f"Starting CLI with log level {args.log_level}")
