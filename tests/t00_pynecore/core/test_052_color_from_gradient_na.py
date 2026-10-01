"""
@pyne

color.from_gradient — the measured TradingView mix, plus the sibling ``color.t()``
accessor.

Every expected color below is a TradingView export (packed bgcolor, CAPITALCOM:EURUSD
1D, 2026-10-01), not a value derived from the implementation. TradingView mixes the
endpoints with their alpha premultiplied and truncates, so a translucent endpoint
contributes less hue, a na endpoint mixes in as the all-zero color, and the
truncation leaves a channel one below the solid endpoint's on some bars. The result
is never na: a na value or bound, equal bounds and an all-transparent mix give the
all-zero color. A reversed range always gives the bottom color.

The na endpoint once raised ``TypeError: NA cannot be converted to int`` (regression:
Volume Order Blocks [BigBeluga], ``color.from_gradient(difff, ..., color(na), col2)``),
and ``color.t()`` once returned the raw 0-255 alpha instead of the 0-100 transparency.
"""
from pynecore.lib import color
from pynecore.types.color import Color
from pynecore.types.na import NA, na_float


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# color.t() returns transparency (0-100), not the raw alpha (0-255)
#

def __test_color_t_returns_transparency_not_alpha__():
    """color.t(opaque) == 0; the old bug returned the 255 alpha."""
    assert color.t(color.red) == 0
    assert color.t(color.red) != 255


def __test_color_t_round_trips_transparency__():
    """color.t of a color built with transparency 40 reads ~40, not the 153 alpha."""
    assert abs(color.t(color.rgb(242, 54, 69, 40)) - 40) <= 1


#
# Mix of two translucent endpoints (premultiplied, truncated)
#

def __test_helper_rgba(c):
    return c.r, c.g, c.b, c.a


def __test_from_gradient_premultiplies_translucent_endpoints__():
    """33% -> 77% transparent: the hue leans toward the more opaque bottom endpoint."""
    bottom = color.new(color.rgb(200, 10, 40), 33)
    top = color.new(color.rgb(20, 90, 250), 77)
    assert __test_helper_rgba(color.from_gradient(0.0, 0, 1, bottom, top)) == (199, 10, 40, 171)
    assert __test_helper_rgba(color.from_gradient(0.25, 0, 1, bottom, top)) == (181, 18, 61, 143)
    assert __test_helper_rgba(color.from_gradient(0.5, 0, 1, bottom, top)) == (153, 30, 93, 115)
    assert __test_helper_rgba(color.from_gradient(1.0, 0, 1, bottom, top)) == (20, 90, 250, 59)


def __test_from_gradient_opaque_endpoints__():
    """Opaque red -> green; the byte alpha mix can land just below 255."""
    v = 159 / 10.0 - 15.0
    assert __test_helper_rgba(color.from_gradient(v, 0, 100, color.red, color.green)) == (240, 55, 69, 254)
    assert __test_helper_rgba(color.from_gradient(50, 0, 100, color.red, color.green)) == (159, 114, 74, 255)


def __test_from_gradient_fully_transparent_endpoint__():
    """A 100%-transparent bottom adds no hue; on its own it mixes to the all-zero color."""
    bottom = color.new(color.blue, 100)
    top = color.new(color.yellow, 0)
    assert __test_helper_rgba(color.from_gradient(0, 0, 100, bottom, top)) == (0, 0, 0, 0)
    assert __test_helper_rgba(color.from_gradient(50, 0, 100, bottom, top)) == (253, 216, 53, 127)


#
# na endpoints mix in as the all-zero color
#

def __test_from_gradient_bottom_na_keeps_top_hue__():
    """Bottom na: the top hue stays (one below on some bars), alpha fades from 0."""
    top = color.new(color.rgb(20, 90, 250), 77)
    assert __test_helper_rgba(color.from_gradient(7 / 1000, 0, 1, NA(Color), top)) == (20, 89, 250, 0)
    assert __test_helper_rgba(color.from_gradient(17 / 1000, 0, 1, NA(Color), top)) == (20, 90, 249, 1)
    assert __test_helper_rgba(color.from_gradient(0.5, 0, 1, NA(Color), top)) == (20, 90, 250, 29)


def __test_from_gradient_top_na_keeps_bottom_hue__():
    """Top na: the bottom hue stays (one below on some bars), alpha fades to 0."""
    bottom = color.new(color.rgb(77, 66, 55), 40)
    v = 151 / 10.0 - 15.0
    assert __test_helper_rgba(color.from_gradient(v, 0, 100, bottom, NA(Color))) == (76, 66, 54, 152)


def __test_from_gradient_at_the_na_endpoint_is_all_zero_not_na__():
    """At the na endpoint's position nothing is left to mix, and the result is not na."""
    result = color.from_gradient(0, 0, 1, NA(Color), color.red)
    assert not isinstance(result, NA)
    assert __test_helper_rgba(result) == (0, 0, 0, 0)
    assert __test_helper_rgba(color.from_gradient(100, 0, 100, color.red, NA(Color))) == (0, 0, 0, 0)
    assert __test_helper_rgba(color.from_gradient(4, 0, 9, NA(Color), NA(Color))) == (0, 0, 0, 0)


#
# Degenerate inputs
#

def __test_from_gradient_na_value_is_all_zero_not_na__():
    """A na driving value gives the all-zero color: TradingView's na() is false on it."""
    result = color.from_gradient(na_float, 0, 9, color.red, color.blue)
    assert not isinstance(result, NA)
    assert __test_helper_rgba(result) == (0, 0, 0, 0)


def __test_from_gradient_equal_bounds_is_all_zero__():
    """Equal bounds give the all-zero color for every value, not the bottom color."""
    for v in (-5, 50, 120):
        assert __test_helper_rgba(color.from_gradient(v, 50, 50, color.red, color.green)) == (0, 0, 0, 0)


def __test_from_gradient_reversed_range_is_the_bottom_color__():
    """bottom_value above top_value gives the bottom color whatever the value."""
    bottom = color.new(color.rgb(1, 2, 3), 90)
    top = color.rgb(254, 128, 7)
    for v in (-15.0, 50.0, 1306 / 10.0 - 15.0):
        assert __test_helper_rgba(color.from_gradient(v, 100, 0, bottom, top)) == (1, 2, 3, 25)
