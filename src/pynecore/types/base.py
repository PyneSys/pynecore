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


# A drawing is a reference type whose identity belongs to a module registry: the
# variable holds a handle, and every ``*.set_*`` call mutates the registered object.
# A field-wise clone carries a DUPLICATE vid (``vid`` is a normal init=True field),
# sits in no registry, never reaches the chart and aliases the original in the viz
# stream, where a linefill serializes its lines by vid. Copying one is only
# expressible through the namespace ``copy()``, which assigns a fresh vid and
# registers the clone.
#
# Measured on TradingView: mutating a drawing through a container copy changes the
# ORIGINAL, for ``matrix.copy()`` and ``array.copy()`` alike — containers share
# their drawing references rather than cloning them.
class Drawing:
    """
    Base of the chart-object types: line, label, box, table, linefill and polyline.
    """

    __slots__ = ()

    def __deepcopy__(self, memo: dict) -> 'Drawing':
        """
        Return the drawing itself: a chart object has no deep copy.

        :param memo: Deepcopy memo, unused
        :return: This same object
        """
        return self


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
