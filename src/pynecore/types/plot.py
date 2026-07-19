from .base import IntEnum


class PlotEnum(IntEnum):
    ...


class Plot:
    __slots__ = ('id',)

    def __init__(self, id: str = ''):  # noqa (shadowing built-in name (id) intentionally)
        self.id = id
