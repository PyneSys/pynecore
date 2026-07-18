"""
IDE-facing view of the color module. Returns are plain ``Color`` (the NA-free
"plain T" policy — see types/persistent.pyi): in Pine semantics any color can
be na, so ``Color | NA[Color]`` returns only poison every downstream callsite.
Parameters stay permissive (accept NA), runtime (color.py) is unchanged.
"""
from ..types.color import Color as Color
from ..types.na import NA

aqua: Color
black: Color
blue: Color
fuchsia: Color
gray: Color
green: Color
lime: Color
maroon: Color
navy: Color
olive: Color
orange: Color
purple: Color
red: Color
silver: Color
teal: Color
white: Color
yellow: Color


def r(color: Color | NA[Color]) -> int: ...


def g(color: Color | NA[Color]) -> int: ...


def b(color: Color | NA[Color]) -> int: ...


def t(color: Color | NA[Color]) -> float: ...


def new(color: Color | str | NA[Color], transp: float | NA[float] = 0) -> Color: ...


def rgb(r: int | float, g: int | float, b: int | float, transp: float = 0) -> Color: ...


def from_gradient(value: int | float | NA[float], bottom_value: int | float | NA[float],
                  top_value: int | float | NA[float],
                  bottom_color: Color | NA[Color], top_color: Color | NA[Color]) -> Color: ...
