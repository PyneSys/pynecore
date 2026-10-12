"""
@pyne

Pine's ``%`` is the truncated remainder, which takes the sign of the dividend.

MEASURED on TradingView (CAPITALCOM:EURUSD 60, last bar), with ``z = bar_index * 0``:
``(z - 7) % 3`` is -1, ``(z + 7) % -3`` is 1, ``(z - 7.5) % 2`` is -1.5 and
``(z - 6.0) % 3`` is -0.0 (``1 / x`` is -inf); ``(z + 7) % 3.5`` is 0,
``(z + 1e17) % 3`` is 1, ``(z - 5.25) % 0.5`` is -0.25 and ``14 / 8 % 1`` is 0.75;
a zero divisor and a na operand give na. Python's floored ``%`` gives 2, -2, 0.5.
"""
from pynecore.lib import bar_index, na, script, string


@script.indicator("Pine Remainder")
def main():
    z = bar_index * 0
    zf = bar_index * 0.0
    nf: float = na
    x: float = 7.0
    x %= zf - 3
    y: float = 7.0
    y /= zf
    w: int = 7
    w %= 2
    return {
        "s1": string.format("{0} {1} {2}", string.tostring((z - 7) % 3), string.tostring((z + 7) % -3),
                            string.tostring((zf - 7.5) % 2)),
        "s2": string.format("{0}", 1 / ((zf - 6.0) % 3) < 0),
        "s3": string.format("{0} {1} {2} {3}", string.tostring((z + 7) % 3.5),
                            string.tostring((zf + 1e17) % 3), string.tostring((zf - 5.25) % 0.5),
                            string.tostring((z + 14) / 8 % 1)),
        "e1": string.format("{0} {1} {2} {3}", na(7 % z), na(7.5 % zf), na(nf % 3), na(7 % nf)),
        "a1": string.format("{0} {1} {2}", string.tostring(x), na(y), string.tostring(w)),
        "py": "%d items" % 3,
    }


def __test_pine_remainder__(runner, dummy_ohlcv_iter):
    """the remainder is truncated, na on a zero divisor or a na operand, also augmented"""
    result = None
    for _candle, plot in runner(dummy_ohlcv_iter).run_iter():
        result = plot
        break
    assert result == {
        "s1": "-1 1 -1.5",
        "s2": "true",
        "s3": "0 1 -0.25 0.75",
        "e1": "true true true true",
        "a1": "1 true 1",
        "py": "3 items",
    }
