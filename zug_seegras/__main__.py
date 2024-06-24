from zug_seegras import getLogger
from zug_seegras import BaseClass

log = getLogger(__name__)
def main() -> str:
    log.info("Starting up")
    base = BaseClass()
    return base.hello_world()

print(main())