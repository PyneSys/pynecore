"""
@pyne

Which digits the number formatters print.

Every expected string below is measured TradingView output for a double computed
at run time (probes m559 and fmt1-fmt4), not inference. Both formatters start from
the digits Java's ``Double.toString`` produces -- the shortest decimal that reads
back as the same double -- instead of the double's exact binary expansion, so
``str.tostring(0.1, '#.####...')`` is ``0.1`` and not ``0.10000000000000000555``.
A mask with enough places prints every one of those digits, at any magnitude:
``0.015721813265222873`` keeps all 18 decimals under a 20-place mask.

A float constant of the script (a literal or a folded constant expression) is not
such a double: TradingView embeds it rounded to 16 decimal places while
``|value| >= 1e-3``, so the literal ``0.30000000000000004`` already arrives as
``0.3``. That rounding happens before the formatter sees the value (it belongs to
``transformers/const_fold.py``); the cases here feed the formatter the double the
TradingView runtime holds.

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
    (0.1 + 0.7, M20, "0.7999999999999999"),
    # ... every one of them, however many decimals that takes
    (0.1 + 0.2, M20, "0.30000000000000004"),
    (0.12345678901234566, M20, "0.12345678901234566"),
    (0.37436195478307144, M20, "0.37436195478307144"),
    (0.37436195478307144, "#.#########################", "0.37436195478307144"),
    (-0.49999999999999983, M20, "-0.49999999999999983"),
    (0.015721813265222873, M20, "0.015721813265222873"),
    (0.009999999999999998, M20, "0.009999999999999998"),
    (0.0037436195478307145, M20, "0.0037436195478307145"),
    (0.001234567890123456, M20, "0.001234567890123456"),
    (0.0012345678901234567, M20, "0.0012345678901234567"),
    (0.0019999999999999996, M20, "0.0019999999999999996"),
    (0.0010000000000000002, M20, "0.0010000000000000002"),
    (0.0009999999999999998, M20, "0.0009999999999999998"),
    (0.00012345678901234567, M20, "0.00012345678901234567"),
    (0.0001234567890123456, M20, "0.0001234567890123456"),
    (1e-17, M20, "0.00000000000000001"),
    (1e-7, M20, "0.0000001"),
    (37436195.47830714, M20, "37436195.47830714"),
    (374361954.78307146, M20, "374361954.78307146"),
    (1234567890123456.8, M20, "1234567890123456.8"),
    (0.37436195478307144, "#.00000000000000000000", ".37436195478307144000"),
    # A mask with fewer places rounds those digits half-up
    (0.1 + 0.2, "#.################", "0.3"),
    (0.015721813265222873, "#.#################", "0.01572181326522287"),
    (0.015721813265222873, "#.################", "0.0157218132652229"),
    (0.0037436195478307145, "#.##################", "0.003743619547830715"),
    (0.0024955535589745548, "#.#################", "0.00249555355897455"),
    (0.22489067655901285, "#.################", "0.2248906765590129"),
    (0.0010000000000000002, "#.##################", "0.001"),
    (-0.49999999999999983, "#.###############", "-0.5"),
    (0.9999999999999999, "#.###############", "1"),
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
    # The same digits as str.tostring, every one of them
    (0.1, "{0,number,#.####################}", "0.1"),
    (1.0 / 3.0, "{0,number,#.####################}", "0.3333333333333333"),
    (0.30000000000000004, "{0,number,#.####################}", "0.30000000000000004"),
    (0.12345678901234566, "{0,number,#.####################}", "0.12345678901234566"),
    (-0.49999999999999983, "{0,number,#.####################}", "-0.49999999999999983"),
    (0.015721813265222873, "{0,number,#.####################}", "0.015721813265222873"),
    (0.001234567890123456, "{0,number,#.####################}", "0.001234567890123456"),
    (0.0037436195478307145, "{0,number,#.#########################}", "0.0037436195478307145"),
    # ... but DecimalFormat rounding: half-even, ties decided on the exact value
    (0.015721813265222873, "{0,number,#.################}", "0.0157218132652229"),
    (0.22489067655901285, "{0,number,#.################}", "0.2248906765590128"),
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
