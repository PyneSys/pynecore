"""
@pyne
"""
from dataclasses import dataclass

import pytest

from pynecore.lib import color
from pynecore.types.color import Color
from pynecore.types.na import NA, na_float


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# Color is a value type (RGBA packed into one int); defining __eq__ without
# __hash__ would make it unhashable, which forbids it as a dataclass/``@udt``
# field default and as a dict key. Both are needed by compiled UDTs whose fields
# default to a color (e.g. ``color = color.blue``).
#

def __test_color_is_hashable_and_consistent_with_eq__():
    """Equal colors hash equal; distinct colors are distinguishable."""
    a = Color('#0000FFFF')
    b = Color('#0000FFFF')
    c = Color('#FF0000FF')
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert {a: 'blue'}[b] == 'blue'


def __test_color_usable_as_dataclass_field_default__():
    """A Color default on a dataclass field is allowed (no mutable-default error)."""
    @dataclass(slots=True)
    class Props:
        col: Color = Color('#0000FFFF')

    assert Props().col == Color('#0000FFFF')


#
# color.new — Pine-compatible NA propagation. TradingView does not raise when the
# transparency (or the base color) is na during warmup; it returns a na color.
#

def __test_color_new_na_transparency_returns_na__():
    """color.new(color, na) yields a na color instead of raising, matching Pine"""
    result = color.new(color.red, na_float)
    assert isinstance(result, NA)


def __test_color_new_na_color_returns_na__():
    """color.new(na, transp) propagates na from the base color"""
    result = color.new(NA(Color), 30)
    assert isinstance(result, NA)


#
# color.new must not mutate its argument — it returns a fresh color
#

def __test_color_new_does_not_mutate_input__():
    """color.new(color.red, 80) leaves the color.red constant untouched"""
    before = color.red.value
    result = color.new(color.red, 80)
    assert color.red.value == before
    assert result is not color.red


#
# The numeric validation path is unchanged for real transparencies
#

def __test_color_new_numeric_path_preserves_rgb__():
    """A valid transparency keeps the base RGB and only adjusts the alpha"""
    result = color.new(Color('#00C3FF'), 0)
    assert result.value == Color('#00C3FF').value


def __test_color_new_out_of_range_still_rejected__():
    """A numeric transparency outside 0..100 is still a ValueError"""
    with pytest.raises(ValueError):
        color.new(Color('#00C3FF'), 150)
