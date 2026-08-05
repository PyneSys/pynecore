"""
@pyne

color.rgb must name its channel arguments the way Pine does (``red``, ``green``,
``blue``), because a compiled script passing them by keyword emits Pine's own names.
A mismatch is a TypeError that halts the script at runtime.
"""
from pynecore.lib import color


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_color_rgb_channel_argument_names__():
    """color.rgb binds red/green/blue by keyword and keeps the positional order."""
    named = color.rgb(red=242, green=54, blue=69, transp=40)
    positional = color.rgb(242, 54, 69, 40)
    assert named == positional
    assert (named.r, named.g, named.b) == (242, 54, 69)
    assert abs(color.t(named) - 40) <= 1
