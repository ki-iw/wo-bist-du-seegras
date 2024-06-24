from zug_seegras import BaseClass


def test_base() -> None:
    base = BaseClass()
    assert isinstance(base, BaseClass)
