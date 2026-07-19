class IntEnum(int):
    """
    IntEnum class that auto-increments values.
    """

    def __init_subclass__(cls, start: int = 0, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = start  # Each subclass gets its own counter

    def __new__(cls):
        # Create new object with the current counter value
        value = cls._counter
        cls._counter += 1
        # noinspection PyTypeChecker
        return super().__new__(cls, value)


class StrLiteral(str):
    """
    StrLiteral class to store string literals.
    """


# Monotonic id counter for drawing objects (line/label/box/table/polyline/linefill).
# Single-threaded runner, so a plain module-level integer needs no locking.
_vid_counter = 0


def next_vid() -> int:
    """Return the next monotonically increasing drawing-object id."""
    global _vid_counter
    _vid_counter += 1
    return _vid_counter


def reset_vid_counter() -> None:
    """Reset the drawing-object id counter (new run / new script)."""
    global _vid_counter
    _vid_counter = 0
