"""
@pyne

The number forms ``str.tostring`` and ``str.format`` print, measured on TradingView.

Every expected string below is TradingView output (CAPITALCOM:EURUSD 60, mintick 0.00001,
probes of 2026-10-11), not inference. The values are run-time doubles on TradingView
(``close * 0.0 + v``), so no compile-time folding is involved.

- ``str.tostring(value)`` without a format is Java's ``DecimalFormat`` with ten optional
  decimals (ties decided on the exact value), not the ``'#.##########'`` mask, which rounds
  half-up; outside ``(1e-21, 1e21)`` it is exponential (``1E21``, ``1.2345678901E-25``).
- ``str.tostring`` never signs a result that has no non-zero digit; ``str.format`` does.
- ``format.mintick`` rounds to whole ticks half away from zero in a Java long and pads to
  the tick's decimals; ``format.percent`` prints the unscaled value with two decimals;
  ``format.volume`` picks K/M/B/T on the raw magnitude and keeps up to three decimals.
- An array prints element by element with the same format, a matrix one row per line.
- ``str.format``'s default number style is ``#,##0.###``, so thousands are grouped, and an
  array argument prints its float elements with Java's ``Double.toString``.

Known residuals, not covered here: the TradingView JVM predates JDK 19, so a few values get
non-shortest ``Double.toString`` digits there (``5e-324`` prints ``4.9E-324``), and
``str.format`` prints an int array element without ``.0`` by its static type, which a
run-time double does not carry.
"""
import pytest

from pynecore.lib import format as _format
from pynecore.lib import syminfo
from pynecore.lib.string import tostring, format as str_format
from pynecore.types.matrix import Matrix


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


@pytest.mark.parametrize("value,expected", [
    (1.0 / 3.0, "0.3333333333"),
    (1e-10, "0.0000000001"),
    # Ties go by the exact value, unlike the explicit '#.##########' mask below
    (2.5e-10, "0.0000000003"),
    (3.5e-10, "0.0000000003"),
    (7.5e-10, "0.0000000007"),
    (1.5e-10, "0.0000000001"),
    (0.12345678925, "0.1234567893"),
    (0.12345678935, "0.1234567893"),
    (0.12345678905, "0.1234567891"),
    (1.00000000005, "1.0000000001"),
    (1234.56789012345, "1234.5678901235"),
    # Java's lone-5 underflow: 5e-11 rounds down, 5.5e-11 and 6e-11 up
    (5e-11, "0"),
    (4e-11, "0"),
    (5.5e-11, "0.0000000001"),
    (6e-11, "0.0000000001"),
    (5.000000000000001e-11, "0.0000000001"),
    (1e-11, "0"),
    (1e-20, "0"),
    (1.5e-21, "0"),
    (1.0000000000000002e-21, "0"),
    (1234567.891, "1234567.891"),
    (123456789012345678.0, "123456789012345680"),
    (9.99e20, "999000000000000000000"),
    (123456789012345678901.0, "123456789012345680000"),
    # Exponential outside (1e-21, 1e21)
    (1e21, "1E21"),
    (1.0000000000000002e21, "1E21"),
    (1.5e21, "1.5E21"),
    (-1e21, "-1E21"),
    (1e22, "1E22"),
    (1.23456789015e21, "1.2345678901E21"),
    (9.99999999999e21, "1E22"),
    (9.999999999949e21, "9.9999999999E21"),
    (1.2345678901234567e25, "1.2345678901E25"),
    (1e300, "1E300"),
    (1.7976931348623157e308, "1.7976931349E308"),
    (1e-21, "1E-21"),
    (9e-22, "9E-22"),
    (1.2345678901234e-25, "1.2345678901E-25"),
    (1e-100, "1E-100"),
    (-1e-300, "-1E-300"),
    (2.2250738585072014e-308, "2.2250738585E-308"),
    (1.5e-320, "1.5E-320"),
    # No sign without a non-zero digit
    (-0.0, "0"),
    (-1e-11, "0"),
    (-0.004, "-0.004"),
])
def __test_tostring_default__(value: float, expected: str):
    """str.tostring without a format"""
    assert tostring(value) == expected


@pytest.mark.parametrize("value,mask,expected", [
    (1e21, "#.##########", "1000000000000000000000"),
    (1e21, "#.##", "1000000000000000000000"),
    (5e-11, "#.##########", "0.0000000001"),
    (3.5e-10, "#.##########", "0.0000000004"),
    (0.12345678935, "#.##########", "0.1234567894"),
    (-0.0, "#.00", ".00"),
    (-0.0, "0.0", "0.0"),
    (-0.0, "#.##", "0"),
    (-0.001, "#.##", "0"),
    (-0.4, "#", "0"),
    (1234567.891, "#,##0.00", "1,234,567.89"),
    (0.0005, "#.###", "0.001"),
])
def __test_tostring_mask__(value: float, mask: str, expected: str):
    """str.tostring with a digit mask"""
    assert tostring(value, mask) == expected


@pytest.mark.parametrize("value,expected", [
    (1.1, "1.10000"),
    (1.07084, "1.07084"),
    (1.234567, "1.23457"),
    (-1.1, "-1.10000"),
    (0.0, "0.00000"),
    (-0.0, "0.00000"),
    (12345.0, "12345.00000"),
    (-0.000001, "0.00000"),
    (0.000005, "0.00001"),
    (0.000015, "0.00002"),
    (0.000025, "0.00003"),
    (-0.000005, "-0.00001"),
    (123456789.123456, "123456789.12346"),
    (1e16, "92233720368547.77000"),
    (1e21, "92233720368547.77000"),
])
def __test_tostring_mintick__(value: float, expected: str):
    """format.mintick rounds to whole ticks and pads to the tick's decimals"""
    saved = syminfo.mintick
    syminfo.mintick = 0.00001
    try:
        assert tostring(value, _format.mintick) == expected
    finally:
        syminfo.mintick = saved


@pytest.mark.parametrize("value,expected", [
    (0.5, "0.50%"),
    (0.123456, "0.12%"),
    (1.0, "1.00%"),
    (-0.25, "-0.25%"),
    (0.0, "0.00%"),
    (-0.0, "0.00%"),
    (0.00004, "0.00%"),
    (12.345678, "12.35%"),
    (0.125, "0.13%"),
    (0.005, "0.01%"),
    (0.015, "0.02%"),
    (1234567.891, "1234567.89%"),
    (-0.001, "0.00%"),
    (100.0, "100.00%"),
    (2.675, "2.68%"),
    (0.0049999, "0.00%"),
    (-0.005, "-0.01%"),
    (1e17, "100000000000000000.00%"),
    (1e21, "1000000000000000000000.00%"),
    (5e-324, "0.00%"),
])
def __test_tostring_percent__(value: float, expected: str):
    """format.percent prints the unscaled value with two decimals"""
    assert tostring(value, _format.percent) == expected


@pytest.mark.parametrize("value,expected", [
    (0.4, "0"),
    (0.5, "1"),
    (1.5, "2"),
    (2.5, "3"),
    (12.345, "12"),
    (999.0, "999"),
    (999.4, "999"),
    (999.5, "1000"),
    (999.6, "1000"),
    (1000.0, "1K"),
    (1234.5, "1.235K"),
    (1234.5678, "1.235K"),
    (1500.0, "1.5K"),
    (1999.5, "2K"),
    (999999.0, "999.999K"),
    (999999.9, "1000K"),
    (1e6, "1M"),
    (1234567.0, "1.235M"),
    (1e9, "1B"),
    (1.5e12, "1.5T"),
    (2e15, "2000T"),
    (2e18, "2000000T"),
    (1e21, "1000000000T"),
    (-0.4, "0"),
    (-0.6, "-1"),
    (-999.5, "-1000"),
    (-1500.0, "-1.5K"),
    (-0.0, "0"),
    (5e-324, "0"),
])
def __test_tostring_volume__(value: float, expected: str):
    """format.volume picks the suffix on the raw magnitude"""
    assert tostring(value, _format.volume) == expected


def __test_tostring_containers__():
    """an array prints element by element, a matrix one row per line"""
    nan = float('nan')
    assert tostring([1.0, 2.5, 3.0]) == "[1, 2.5, 3]"
    assert tostring([1, 2, 3]) == "[1, 2, 3]"
    assert tostring([nan, nan]) == "[NaN, NaN]"
    assert tostring([0.1 + 0.2, 1e-7, 123456789.123, -0.0, 1e21, 1.0 / 3.0]) == \
           "[0.3, 0.0000001, 123456789.123, 0, 1E21, 0.3333333333]"
    assert tostring([True, False]) == "[true, false]"
    assert tostring(["a", "b c"]) == "[a, b c]"
    assert tostring([]) == "[]"
    assert tostring([-3, 1000000]) == "[-3, 1000000]"
    assert tostring([1.23456, 2.0], "#.##") == "[1.23, 2]"
    assert tostring([1e21], "#.##########") == "[1000000000000000000000]"
    assert tostring([0.5, 1500.0], _format.volume) == "[1, 1.5K]"
    assert tostring([0.5], _format.percent) == "[0.50%]"
    assert tostring(Matrix(2, 2, 1.5)) == "[1.5, 1.5]\n[1.5, 1.5]"
    assert tostring(Matrix(2, 3, 7)) == "[7, 7, 7]\n[7, 7, 7]"
    assert tostring(Matrix(1, 2, True)) == "[true, true]"
    assert tostring(Matrix(1, 2, "x")) == "[x, x]"
    assert tostring(Matrix(0, 0)) == ""


def __test_tostring_ignores_the_python_number_class__():
    """an int and the equal double print alike, inside a container too"""
    assert tostring(7) == tostring(7.0) == "7"
    assert tostring([0, 2, 1.0]) == tostring([0.0, 2.0, 1.0]) == "[0, 2, 1]"
    assert tostring(0) == tostring(-0.0) == "0"


@pytest.mark.parametrize("pattern,value,expected", [
    ("{0}", 1234.5678, "1,234.568"),
    ("{0}", -1234.5678, "-1,234.568"),
    ("{0}", 1234567.0, "1,234,567"),
    ("{0}", 1234567, "1,234,567"),
    ("{0}", 1e21, "1,000,000,000,000,000,000,000"),
    ("{0}", 123456789012345678.0, "123,456,789,012,345,680"),
    ("{0}", -0.0, "-0"),
    ("{0}", 1.0, "1"),
    ("{0}", 7, "7"),
    ("{0}", 2.5, "2.5"),
    ("{0}", 999.9996, "1,000"),
    ("{0}", 0.0015, "0.002"),
    ("{0}", 0.0025, "0.003"),
    ("{0}", 0.0005, "0"),
    ("{0}", 0.00051, "0.001"),
    ("{0}", 0.0006, "0.001"),
    ("{0}", 1e-300, "0"),
    ("{0,number}", 1234.5678, "1,234.568"),
    ("{0,number,#,##0.00}", 1234567.891, "1,234,567.89"),
    ("{0,number,integer}", 1234567.5, "1,234,568"),
    ("{0,number,percent}", 1234.5, "123,450%"),
    ("{0,number,#.#}", 0.05, "0.1"),
    ("{0,number,#.#}", 0.25, "0.2"),
    ("{0,number,#.##}", -0.0, "-0"),
    ("{0,number,#.##}", -0.001, "-0"),
    ("{0,number,0.00}", -0.0, "-0.00"),
    ("{0,number,#.##########}", 5e-11, "0"),
])
def __test_str_format_numbers__(pattern: str, value: float, expected: str):
    """str.format's DecimalFormat forms"""
    assert str_format(pattern, value) == expected


def __test_str_format_arrays__():
    """an array argument prints its elements' own toString"""
    nan = float('nan')
    assert str_format("{0}", [1.0, 2.5, 1234.5678, 1e21, -0.0, 1.0 / 3.0, 0.0005]) == \
           "[1.0, 2.5, 1234.5678, 1.0E21, -0.0, 0.3333333333333333, 5.0E-4]"
    assert str_format("{0}", [1, -3, 1000000]) == "[1, -3, 1000000]"
    assert str_format("{0}", [True, False]) == "[true, false]"
    assert str_format("{0}", ["a", "b c"]) == "[a, b c]"
    assert str_format("{0}", [nan, nan]) == "[NaN, NaN]"
