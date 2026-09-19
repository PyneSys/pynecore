"""
@pyne

The ``str.tostring`` fast path must agree with the general formatter everywhere.

``_format_number`` routes a finite double with a digit mask or ``format.mintick``
through ``_format_parsed``, which parses the mask once per pattern string and rounds
half-up on the digits of ``repr`` instead of building a ``Decimal`` per call.
``_format_number_general`` stays in the code as the fallback and as the semantic
reference, so the two can be compared directly: every case below asserts the fast
path produces the very same string the general path does.

The value grid is deterministic (seeded) and deliberately loaded with the shapes
where a digit-string rounder could drift from ``Decimal.quantize``: exact half-way
digits, carries across the decimal point, the 16-fraction-digit cap at
``|value| >= 1e-3``, the boundaries where ``repr`` switches to exponential notation
(below 1e-4 and from 1e16), and negative zero.
"""
import random

import pytest

from pynecore.lib import format as _format
from pynecore.lib import syminfo
# noinspection PyProtectedMember
from pynecore.lib.string import _format_number, _format_number_general, tostring
from pynecore.lib.string import format as pine_format

__test_helper_MASKS = (
    '#.##########', '#', '#.#', '#.##', '#.###', '#.####', '0', '0.0', '0.00', '#.00',
    '#.0#', '00.000', '000', '#.####################', '0.00000000000000000000',
    '$#.##', '#.##%', '+#.##;-#.##', '#.##;(#.##)', '#,##0.00', '0.00####',
)

__test_helper_MINTICKS = (0.01, 0.25, 0.5, 1.0, 0.00001, 1e-8, 5e-05)

# Values whose shortest digits sit exactly on a rounding tie, or on one of the
# boundaries where the digit handling changes shape.
__test_helper_EDGE_VALUES = (
    0.0, -0.0, 2.675, 1.005, 0.125, 0.375, 9.995, 99.9995, 0.0005, 0.00049999,
    0.5, 1.5, 2.5, 9.999, 99.99999999, 0.1, 0.1 + 0.2, 0.1 + 0.7, 1.0 / 3.0,
    1e-4, 1e-3, 9.999999e-4, 1e15, 1e16, 1e17, 5e-324, 1e-17,
    123456789.123456789, 0.001234567890123456, 1234.5, 1200.0, 5.0,
)


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass

# Values that sit exactly on a rounding tie in binary, where half-even decides
__test_helper_BINARY_TIES = [sign * n / 2 ** k for sign in (1.0, -1.0)
                             for k in range(1, 9) for n in (1, 3, 5, 7, 21, 1001, 123457)]


def __test_helper_grid():
    """Build the deterministic value grid the comparison runs over."""
    rng = random.Random(20260919)
    values = list(__test_helper_EDGE_VALUES)
    for value in __test_helper_EDGE_VALUES:
        values.append(-value)
    for _ in range(400):
        values.append(round(rng.uniform(-100000, 100000), rng.randint(0, 10)))
        values.append(rng.uniform(-1, 1) * 10 ** rng.randint(-12, 17))
        values.append(float(rng.randint(-10 ** 15, 10 ** 15)))
    return values


@pytest.mark.parametrize("mask", __test_helper_MASKS)
def __test_fast_path_matches_general_for_masks__(mask: str):
    """every digit mask formats the same through both paths"""
    for value in __test_helper_grid():
        assert _format_number(value, precision=mask) == \
               _format_number_general(value, precision=mask), (value, mask)


@pytest.mark.parametrize("mask", __test_helper_MASKS)
def __test_fast_path_matches_general_for_str_format__(mask: str):
    """str.format rounds on the exact binary value; both paths must agree there too"""
    for value in __test_helper_grid() + __test_helper_BINARY_TIES:
        assert _format_number(value, precision=mask, decimal_format=True) == \
               _format_number_general(value, precision=mask, decimal_format=True), (value, mask)


@pytest.mark.parametrize("value,pattern,expected", [
    (2.675, '{0,number,#.##}', '2.67'),
    (0.125, '{0,number,#.##}', '0.12'),
    (0.375, '{0,number,#.##}', '0.38'),
    (2.5, '{0,number,#}', '2'),
    (3.5, '{0,number,#}', '4'),
    (1234.5678, '{0}', '1234.568'),
    (0.5, '{0,number,#.0}', '.5'),
    (7.0, '{0,number,0.00}', '7.00'),
])
def __test_str_format_literal_expectations__(value: float, pattern: str, expected: str):
    """exact ties go to even and inexact ones follow the double's true value"""
    assert pine_format(pattern, value) == expected


@pytest.mark.parametrize("tick", __test_helper_MINTICKS)
def __test_fast_path_matches_general_for_mintick__(tick: float):
    """format.mintick formats the same through both paths, for every tick size"""
    saved = syminfo.mintick
    syminfo.mintick = tick
    try:
        for value in __test_helper_grid():
            assert _format_number(value, fmt_type=_format.mintick) == \
                   _format_number_general(value, fmt_type=_format.mintick), (value, tick)
    finally:
        syminfo.mintick = saved


def __test_mintick_follows_the_current_symbol__():
    """a tick size change takes effect immediately -- nothing caches a formatted result"""
    saved = syminfo.mintick
    try:
        syminfo.mintick = 0.01
        assert _format_number(1234.5678, fmt_type=_format.mintick) == '1234.57'
        syminfo.mintick = 0.00001
        assert _format_number(1234.5678, fmt_type=_format.mintick) == '1234.5678'
        syminfo.mintick = 1.0
        assert _format_number(1234.5678, fmt_type=_format.mintick) == '1235'
    finally:
        syminfo.mintick = saved


def __test_malformed_unused_subpattern_is_ignored__():
    """only the subpattern the sign selects is parsed: a broken other side is harmless"""
    assert tostring(1.5, '#.##;#..##') == '1.5'
    assert tostring(-1.5, '#..##;#.##') == '1.5'
    for value, mask in ((-1.5, '#.##;#..##'), (1.5, '#..##;#.##')):
        with pytest.raises(ValueError):
            _format_number_general(value, precision=mask)
        with pytest.raises(ValueError):
            tostring(value, mask)


def __test_native_float_dispatch__():
    """a native double, na included, formats like any other number"""
    nan = float('nan')
    assert tostring(nan) == 'NaN'
    assert tostring(nan, _format.percent) == 'NaN%'
    assert tostring(float('inf'), _format.mintick) == 'Infinity'
    assert tostring(2.5, _format.percent) == _format_number_general(2.5, fmt_type=_format.percent)
    assert tostring(3.0) == tostring(3) == '3'


@pytest.mark.parametrize("value,mask,expected", [
    (2.675, '#.##', '2.68'),
    (0.1, '#.####################', '0.1'),
    (0.1 + 0.2, '#.####################', '0.3'),
    (123456789.123456789, '#.##########', '123456789.12345679'),
    (9.999, '#', '10'),
    (1200.0, '#.##########', '1200'),
    (0.5, '#.0', '.5'),
    (0.5, '#.#', '0.5'),
    (-0.0, '#.##', '-0'),
    (-0.001, '#.##', '-0'),
    (5.0, '0.00####', '5'),
    (-3.5, '+#.##;-#.##', '-3.5'),
    (-3.5, '#.##;(#.##)', '(3.5)'),
    (12.3456, '$#.##', '$12.35'),
    (7.5, '000', '008'),
])
def __test_literal_expectations__(value: float, mask: str, expected: str):
    """the fast path keeps the strings the formatter is documented to produce"""
    assert tostring(value, mask) == expected
