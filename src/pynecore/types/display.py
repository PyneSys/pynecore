from .base import BitSet


class Display(BitSet):
    """
    A set of chart places a plot or an input shows in: one bit per place, and
    ``display.all`` with every bit set. ``display.all - display.pane`` keeps every
    other place.
    """

    __slots__ = ()
