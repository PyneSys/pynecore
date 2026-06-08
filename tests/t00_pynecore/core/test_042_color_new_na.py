"""
@pyne
"""
import pytest

from pynecore.lib import color
from pynecore.types.color import Color
from pynecore.types.na import NA, na_float


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


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
