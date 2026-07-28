"""
@pyne

Formatting a non-finite number must never raise.

``safe_div`` returns a native ``inf`` on division by zero, and a Pine script
routinely feeds that straight into ``str.tostring`` (``(close - entry) / entry``
before the first entry is set). Infinity used to fall through to the digit
patterns, where ``round()`` raises ``OverflowError`` and ``Decimal`` raises
``InvalidOperation`` -- a formatting call killed the whole run.

The expected strings below are measured TradingView output, not inference. TV
runs two different number formatters: ``str.tostring`` uses the chart's number
formatter (``NaN`` everywhere except ``format.mintick``, which stringifies the
raw double), while ``str.format`` is Java DecimalFormat (the infinity symbol
carries the pattern's prefix and suffix, NaN does not).
"""
import pytest

from pynecore.lib import format as _format
# noinspection PyProtectedMember
from pynecore.lib.string import _format_number

INF = float('inf')
NEG_INF = float('-inf')
NAN = float('nan')


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


@pytest.mark.parametrize("value,fmt_type,expected", [
    (INF, _format.mintick, "Infinity"),
    (NEG_INF, _format.mintick, "-Infinity"),
    (NAN, _format.mintick, "NaN"),
    (INF, '', "NaN"),
    (INF, _format.percent, "NaN%"),
    (INF, _format.volume, "NaNT"),
    (INF, _format.price, "NaN"),
    (INF, _format.inherit, "NaN"),
    (NAN, _format.percent, "NaN%"),
    (NAN, _format.volume, "NaN"),
])
def __test_tostring_non_finite__(value: float, fmt_type: str, expected: str):
    """str.tostring's chart formatter never raises on inf/nan"""
    assert _format_number(value, fmt_type=fmt_type) == expected


@pytest.mark.parametrize("value,fmt_type,precision,expected", [
    (INF, '', '#.###', "∞"),
    (NEG_INF, '', '#.###', "-∞"),
    (NAN, '', '#.###', "NaN"),
    (INF, 'integer', '#.###', "∞"),
    (INF, 'currency', '#.###', "$∞"),
    (INF, _format.percent, '#.###', "∞%"),
    (INF, '', '+#.##;-#.##', "+∞"),
    (NAN, 'currency', '#.###', "NaN"),
])
def __test_format_non_finite__(value: float, fmt_type: str, precision: str, expected: str):
    """str.format's DecimalFormat affixes the infinity symbol but not NaN"""
    assert _format_number(value, fmt_type=fmt_type, precision=precision,
                          decimal_format=True) == expected


def __test_finite_values_untouched__(log):
    """the non-finite branch does not disturb ordinary numbers"""
    from pynecore.lib import syminfo

    syminfo.mintick = 0.01
    assert _format_number(1234.5678, fmt_type=_format.mintick) == '1234.57'
    assert _format_number(0.25, fmt_type=_format.percent) == '25%'
    assert _format_number(-3.5, precision='+#.##;-#.##') == '-3.5'
