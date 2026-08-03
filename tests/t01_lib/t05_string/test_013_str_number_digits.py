"""
@pyne

Which digits the number formatters print.

Every expected string below is measured TradingView output (probes m559), not
inference. Both formatters start from the digits Java's ``Double.toString``
produces -- the shortest decimal that reads back as the same double -- instead
of the double's exact binary expansion, so ``str.tostring(0.1, '#.####...')``
is ``0.1`` and not ``0.10000000000000000555``. While that representation stays
in plain notation (``|value| >= 1e-3``) it never carries more than 16 fraction
digits; below 1e-3 ``Double.toString`` switches to exponential notation and the
cap disappears.

The two formatters differ in how they round: ``str.tostring`` is half-up over
those digits, ``str.format`` is Java DecimalFormat, which resolves a tie on the
exact value (2.675 is really 2.67499..., so it goes down where ``str.tostring``
goes up).

Known limit: the TradingView JVM predates JDK 19, whose ``Double.toString``
gained the shortest-representation guarantee, so a handful of values print an
extra digit there (``1e23`` is ``9.999999999999999E22`` on that JVM). Python's
``repr`` is always shortest, so those cases -- only reachable above 1e17 or
through a full-precision mask -- still differ.
"""
import pytest

from pynecore.lib.string import tostring, format as str_format

M20 = "#.####################"


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


@pytest.mark.parametrize("value,fmt,expected", [
    # Shortest round-trip digits, not the exact binary expansion
    (0.1, M20, "0.1"),
    (1.0 / 3.0, M20, "0.3333333333333333"),
    (-36.51483716701107, "#.################", "-36.51483716701107"),
    (123456789.123456789, "#.##########", "123456789.12345679"),
    (12345.678901234567891, M20, "12345.678901234567"),
    # The 16-decimal cap in the plain-decimal range
    (0.1 + 0.2, M20, "0.3"),
    (0.1 + 0.2, "#.################", "0.3"),
    (0.001234567890123456, M20, "0.0012345678901235"),
    (0.0012345678901234567, M20, "0.0012345678901235"),
    (0.009999999999999998, M20, "0.01"),
    (0.0019999999999999996, M20, "0.002"),
    (0.1 + 0.7, M20, "0.7999999999999999"),
    # ... which is gone below 1e-3, where Double.toString goes exponential
    (0.00012345678901234567, M20, "0.00012345678901234567"),
    (0.0001234567890123456, M20, "0.0001234567890123456"),
    (0.0009999999999999998, M20, "0.0009999999999999998"),
    (1e-17, M20, "0.00000000000000001"),
    (1e-7, M20, "0.0000001"),
    # Half-up rounding over those digits
    (2.675, "#.##", "2.68"),
    (2.675, "#.00", "2.68"),
    (4.35, "#.#", "4.4"),
    (-0.25, "#.#", "-0.3"),
    (0.5, "#", "1"),
    (1.5, "#", "2"),
    (2.5, "#", "3"),
    (123.456, "#", "123"),
    # A mask with required decimals drops a zero integer part, an all-'#' one keeps it
    (0.5, "#.#", "0.5"),
    (0.5, "#.0", ".5"),
    (-0.5, "#.0", "-.5"),
    (0.05, "#.0", ".1"),
    (0.5, "0.#", "0.5"),
    (0.5, "0.0", "0.5"),
    (0.0, "#.0", ".0"),
    (123.0, "#.0", "123.0"),
    (0.1, "#.00000000000000000000", ".10000000000000000000"),
    (0.0, "#.##", "0"),
    (0.0, "#", "0"),
    # Magnitudes far outside the mask
    (1e21, "#.##", "1000000000000000000000"),
    (1e100, "#.##", "1" + "0" * 100),
    (99999.99999999999999, M20, "100000"),
])
def __test_tostring_digits__(value: float, fmt: str, expected: str):
    """str.tostring prints TradingView's digits"""
    assert tostring(value, fmt) == expected


@pytest.mark.parametrize("value,fmt,expected", [
    # Same digits and the same cap as str.tostring
    (0.1, "{0,number,#.####################}", "0.1"),
    (1.0 / 3.0, "{0,number,#.####################}", "0.3333333333333333"),
    (0.30000000000000004, "{0,number,#.####################}", "0.3"),
    (0.001234567890123456, "{0,number,#.####################}", "0.0012345678901235"),
    (0.12345678901234567, "{0,number,#.####################}", "0.1234567890123457"),
    # ... but DecimalFormat rounding: half-even, ties decided on the exact value
    (2.675, "{0,number,#.##}", "2.67"),
    (0.5, "{0,number,#}", "0"),
    (2.5, "{0,number,#}", "2"),
    (0.75, "{0,number,#.#}", "0.8"),
    (0.5, "{0,number,#.#}", "0.5"),
    (0.5, "{0,number,#.0}", ".5"),
    (0.05, "{0,number,#.#}", "0.1"),
])
def __test_format_digits__(value: float, fmt: str, expected: str):
    """str.format prints TradingView's digits"""
    assert str_format(fmt, value) == expected
