from typing import cast

from ..types.color import Color
from ..types.na import NA
from ..types.pine_types import PyneFloat

#
# Constants
#

aqua = Color('#00BCD4')
black = Color('#363A45')
blue = Color('#2962ff')
fuchsia = Color('#E040FB')
gray = Color('#787B86')
green = Color('#4CAF50')
lime = Color('#00E676')
maroon = Color('#880E4F')
navy = Color('#311B92')
olive = Color('#808000')
orange = Color('#FF9800')
purple = Color('#9C27B0')
red = Color('#F23645')
silver = Color('#B2B5BE')
teal = Color('#089981')
white = Color('#FFFFFF')
yellow = Color('#FDD835')


def r(color: Color) -> PyneFloat:
    """
    Return the red component of a color

    :param color: Color
    :return: The red component of the color
    """
    # The component of a na color is 0
    return float(0 if isinstance(color, NA) else color.r)


def g(color: Color) -> PyneFloat:
    """
    Return the green component of a color

    :param color: Color
    :return: The green component of the color
    """
    # The component of a na color is 0
    return float(0 if isinstance(color, NA) else color.g)


def b(color: Color) -> PyneFloat:
    """
    Return the blue component of a color

    :param color: Color
    :return: The blue component of the color
    """
    # The component of a na color is 0
    return float(0 if isinstance(color, NA) else color.b)


def t(color: Color) -> PyneFloat:
    """
    Return the transparency of a color

    :param color: Color
    :return: The transparency of the color, 0-100 (0: not transparent, 100: invisible)
    """
    # A na color is fully transparent. Otherwise the transparency is read back from the
    # alpha byte and rounded to a whole number: color.new(c, 50.3) gives 50, 0.4 gives 0
    # (no tie is possible, 100 * alpha / 255 never ends in .5)
    return 100.0 if isinstance(color, NA) else float(round(color.t))


# noinspection PyShadowingNames
def new(color: Color | str | NA[Color], transp: float | NA[float] = 0) -> Color | NA[Color]:
    """
    Return a new color with the same RGB values and a different transparency

    :param color: A color object or a string in "#RRGGBB" or "#RRGGBBAA" format
    :param transp: Transparency percentage (0-100, 0: not transparent, 100: invisible)
    """
    # Pine propagates na: a na color or na transparency yields a na color
    if isinstance(color, NA) or not (transp == transp):
        return NA(Color)
    if isinstance(color, str):
        color = Color(color)
    # Build a fresh color so the caller's color (e.g. a color.* constant) is not mutated
    result = Color(f'#{color.value:08X}')
    # The guard above rules out na, which the positive na test cannot narrow away
    result.t = cast(float, transp)
    return result


# noinspection PyShadowingNames
def rgb(red: float | NA[float], green: float | NA[float], blue: float | NA[float],
        transp: float | NA[float] = 0) -> Color:
    """
    Return a new color with the given RGB values and transparency

    Fractional values are truncated, out of range values are clipped, and an na value
    counts as 0 for a channel and as fully transparent for the transparency.

    :param red: Red value (0-255)
    :param green: Green value (0-255)
    :param blue: Blue value (0-255)
    :param transp: Transparency percentage (0-100, 0: not transparent, 100: invisible)
    """
    return Color.rgb(red, green, blue, transp)


def from_gradient(value: int | float | NA[float], bottom_value: int | float | NA[float],
                  top_value: int | float | NA[float],
                  bottom_color: Color | NA[Color], top_color: Color | NA[Color]) -> Color:
    """
    Based on the relative position of value in the bottom_value to top_value range,
    the function returns a color from the gradient defined by bottom_color to top_color.

    The colors are mixed with their alpha premultiplied, so a more transparent endpoint
    contributes less hue. The result is never na: where TradingView has nothing to mix
    (a na value or bound, equal bounds, or two fully transparent endpoints) it returns
    the all-zero color, transparent with no hue.

    :param value: Value to calculate the position-dependent color
    :param bottom_value: Bottom position value corresponding to bottom_color
    :param top_value: Top position value corresponding to top_color
    :param bottom_color: Bottom position color
    :param top_color: Top position color
    :return: A color calculated from the linear gradient between bottom_color to top_color
    """
    # Measured on TradingView (CAPITALCOM:EURUSD 1D, 9844 packed bgcolor exports over
    # opaque, translucent, fully transparent and na endpoints, bit-exact): the value is
    # clamped into the range as max(min(value, top), bottom) -- so a reversed range
    # always lands on the bottom color -- and every step below runs as written, with the
    # Java (int) cast that turns the NaN of a na operand or of a 0/0 into 0. A na color
    # mixes in as the all-zero color.
    if not (value == value and bottom_value == bottom_value and top_value == top_value):
        return Color('#00000000')
    span = top_value - bottom_value
    if span == 0:
        return Color('#00000000')
    position = (max(min(value, top_value), bottom_value) - bottom_value) / span
    rest = 1 - position

    bottom = 0 if isinstance(bottom_color, NA) else bottom_color.value
    top = 0 if isinstance(top_color, NA) else top_color.value
    bottom_a = bottom & 0xFF
    top_a = top & 0xFF
    # The alpha mixes as a byte, and the premultiplied channels are divided by that
    # mixed byte scaled back to 0-1 -- not by the mix of the 0-1 alphas, which rounds
    # differently and moves a channel by one on about a quarter of the bars
    alpha = bottom_a * rest + top_a * position
    if alpha == 0:
        return Color('#00000000')
    scale = alpha / 255
    bottom_w = bottom_a / 255
    top_w = top_a / 255
    result = 0
    for shift in (24, 16, 8):
        mixed = ((bottom_w * ((bottom >> shift) & 0xFF)) * rest
                 + (top_w * ((top >> shift) & 0xFF)) * position)
        result |= int(mixed / scale) << shift
    return Color(f'#{result | int(alpha):08X}')
