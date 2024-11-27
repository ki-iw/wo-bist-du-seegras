from zug_seegras import BaseClass, getLogger

log = getLogger(__name__)


def main() -> str:
    log.info("Starting up")
    base = BaseClass()
    return base.hello_world()


print(main())
