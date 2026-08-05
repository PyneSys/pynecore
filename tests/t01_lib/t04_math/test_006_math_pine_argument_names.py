"""
@pyne

The math namespace must name its arguments exactly as Pine does, because a compiled
script that passes them by keyword emits Pine's own name. A mismatch is a TypeError
that halts the script at runtime.
"""
import math as _pymath

from pynecore.lib import math


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def __test_math_inverse_trig_argument_is_named_angle__():
    """math.acos/asin/atan take ``angle``, not ``value``."""
    assert math.acos(angle=0.5) == _pymath.acos(0.5)
    assert math.asin(angle=0.5) == _pymath.asin(0.5)
    assert math.atan(angle=0.5) == _pymath.atan(0.5)


def __test_math_angle_conversion_argument_names__():
    """math.todegrees takes ``radians`` and math.toradians takes ``degrees``."""
    assert math.todegrees(radians=_pymath.pi) == 180.0
    assert math.toradians(degrees=180.0) == _pymath.pi
