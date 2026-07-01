"""
@pyne

color.from_gradient — Pine-compatible handling of a na color endpoint, plus na
propagation through the value/bounds. TradingView (verified with a probe script
exporting color.r/g/b/t of the result on BINANCE:BTCUSDT) does NOT fade the
visible RGB toward transparent-black when one gradient endpoint is na: it keeps
the solid endpoint's RGB and only fades the transparency toward the na side (a
na color reads as r=g=b=0, transparency=100), returning the literal na color
exactly at the na endpoint's position. The old implementation ran a bare
``int()`` over the na channels and raised ``TypeError: NA cannot be converted to
int`` (regression: Volume Order Blocks [BigBeluga], `color.from_gradient(difff,
..., color(na), col2)`).

Also covers the sibling ``color.t()`` accessor, which returned the raw 0-255
alpha instead of the 0-100 transparency the docstring (and Pine) promise.
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
# from_gradient with a na endpoint — solid RGB kept, transparency faded
#

def __test_from_gradient_bottom_na_interior_keeps_top_rgb__():
    """Bottom endpoint na: interior keeps top RGB, transparency fades 100 -> top."""
    result = color.from_gradient(1, 0, 9, NA(Color), color.red)  # position 1/9
    assert not isinstance(result, NA)
    assert (result.r, result.g, result.b) == (242, 54, 69)
    assert abs(result.t - (100.0 * (1 - 1 / 9))) <= 1.5           # ~88.9


def __test_from_gradient_top_na_interior_keeps_bottom_rgb__():
    """Top endpoint na: interior keeps bottom RGB, transparency fades bottom -> 100."""
    result = color.from_gradient(1, 0, 9, color.red, NA(Color))  # position 1/9
    assert not isinstance(result, NA)
    assert (result.r, result.g, result.b) == (242, 54, 69)
    assert abs(result.t - (100.0 * (1 / 9))) <= 1.5               # ~11.1


def __test_from_gradient_bottom_na_at_endpoint_is_na__():
    """At the na endpoint's exact position the result is the literal na color."""
    assert isinstance(color.from_gradient(0, 0, 9, NA(Color), color.red), NA)


def __test_from_gradient_top_na_at_endpoint_is_na__():
    """Symmetric: value at the top na endpoint yields na."""
    assert isinstance(color.from_gradient(9, 0, 9, color.red, NA(Color)), NA)


def __test_from_gradient_both_na_is_na__():
    """Two na endpoints have no hue to keep -> na."""
    assert isinstance(color.from_gradient(4, 0, 9, NA(Color), NA(Color)), NA)


#
# na value / bounds propagate to a na color (Pine na propagation)
#

def __test_from_gradient_na_value_propagates__():
    """A na driving value yields a na color, matching TradingView warmup bars."""
    assert isinstance(color.from_gradient(na_float, 0, 9, color.red, color.blue), NA)


#
# Two solid endpoints keep the original linear interpolation (no regression)
#

def __test_from_gradient_solid_endpoints_interpolate__():
    """Midpoint of red -> blue interpolates each RGB channel linearly."""
    result = color.from_gradient(4.5, 0, 9, color.red, color.blue)  # position 0.5
    assert not isinstance(result, NA)
    assert result.r == int(242 + (41 - 242) * 0.5)
    assert result.g == int(54 + (98 - 54) * 0.5)
    assert result.b == int(69 + (255 - 69) * 0.5)
