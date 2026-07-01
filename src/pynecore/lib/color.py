from ..types.color import Color
from ..types.na import NA

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


def r(color: Color) -> int:
    """
    Return the red component of a color

    :param color: Color
    :return: The red component of the color
    """
    return color.r


def g(color: Color) -> int:
    """
    Return the green component of a color

    :param color: Color
    :return: The green component of the color
    """
    return color.g


def b(color: Color) -> int:
    """
    Return the blue component of a color

    :param color: Color
    :return: The blue component of the color
    """
    return color.b


def t(color: Color) -> float:
    """
    Return the transparency of a color

    :param color: Color
    :return: The transparency of the color, 0-100 (0: not transparent, 100: invisible)
    """
    return color.t


# noinspection PyShadowingNames
def new(color: Color | str | NA[Color], transp: float | NA[float] = 0) -> Color | NA[Color]:
    """
    Return a new color with the same RGB values and a different transparency

    :param color: A color object or a string in "#RRGGBB" or "#RRGGBBAA" format
    :param transp: Transparency percentage (0-100, 0: not transparent, 100: invisible)
    """
    # Pine propagates na: a na color or na transparency yields a na color
    if isinstance(color, NA) or isinstance(transp, NA):
        return NA(Color)
    if isinstance(color, str):
        color = Color(color)
    # Build a fresh color so the caller's color (e.g. a color.* constant) is not mutated
    result = Color(f'#{color.value:08X}')
    result.t = transp
    return result


# noinspection PyShadowingNames
def rgb(r: int, g: int, b: int, transp: float = 0) -> Color:
    """
    Return a new color with the given RGB values and transparency

    :param r: Red value
    :param g: Green value
    :param b: Blue value
    :param transp: Transparency percentage (0-100, 0: not transparent, 100: invisible)
    """
    return Color.rgb(r, g, b, transp)


def from_gradient(value: int | float | NA[float], bottom_value: int | float | NA[float],
                  top_value: int | float | NA[float],
                  bottom_color: Color | NA[Color], top_color: Color | NA[Color]) -> Color | NA[Color]:
    """
    Based on the relative position of value in the bottom_value to top_value range,
    the function returns a color from the gradient defined by bottom_color to top_color.

    :param value: Value to calculate the position-dependent color
    :param bottom_value: Bottom position value corresponding to bottom_color
    :param top_value: Top position value corresponding to top_color
    :param bottom_color: Bottom position color
    :param top_color: Top position color
    :return: A color calculated from the linear gradient between bottom_color to top_color
    """
    # na value/bounds propagate: TradingView returns a na color for such a bar.
    if isinstance(value, NA) or isinstance(bottom_value, NA) or isinstance(top_value, NA):
        return NA(Color)

    # Handle edge cases
    if top_value == bottom_value:
        return bottom_color

    # Calculate the position as a ratio (0.0 to 1.0), clamped to [0, 1]
    position = (value - bottom_value) / (top_value - bottom_value)
    position = max(0.0, min(1.0, position))

    # A na endpoint carries no hue (TV-verified: color.r/g/b(na)==0, color.t(na)==100).
    # TradingView does NOT fade the visible RGB toward transparent-black; it keeps the
    # solid endpoint's RGB and only fades transparency toward the na side (na transp=100),
    # returning the literal na color exactly at the na endpoint's position.
    bottom_na = isinstance(bottom_color, NA)
    top_na = isinstance(top_color, NA)
    if bottom_na and top_na:
        return NA(Color)
    if bottom_na:
        if position <= 0.0:
            return NA(Color)
        transp = 100.0 + (top_color.t - 100.0) * position
        return Color.rgb(top_color.r, top_color.g, top_color.b, transp)
    if top_na:
        if position >= 1.0:
            return NA(Color)
        transp = bottom_color.t + (100.0 - bottom_color.t) * position
        return Color.rgb(bottom_color.r, bottom_color.g, bottom_color.b, transp)

    # Both endpoints solid: interpolate RGB and transparency
    red_comp = int(bottom_color.r + (top_color.r - bottom_color.r) * position)
    green_comp = int(bottom_color.g + (top_color.g - bottom_color.g) * position)
    blue_comp = int(bottom_color.b + (top_color.b - bottom_color.b) * position)
    transp = bottom_color.t + (top_color.t - bottom_color.t) * position

    return Color.rgb(red_comp, green_comp, blue_comp, transp)
