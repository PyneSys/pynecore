"""
@pyne
"""
from pynecore.lib import script, close, ta

# TradingView's `int / int` is int-TYPED while keeping its fractional VALUE, so an
# int parameter can legally receive a float. Measured on BINANCE:BTCUSDT 30m, in
# Pine v4 and v6 alike, with `R = input.int(14)`:
#
#     R / 8            -> 1.75          (the value is NOT truncated)
#     R / 8 * 100      -> 175           (arithmetic keeps the fraction)
#     ta.highest(R / 8)-> ta.highest(1) (truncated where an integer is required)
#     ta.sma(close, R / 8) -> ta.sma(close, 1)
#     close[R / 8]     -> close[1]
#
# `ta.highest(R / 8.0)` is a TradingView COMPILE error ("input float ... should be
# of type: integer"), which is what makes the accepted form a type-system quirk
# rather than a general float length: only an int-typed expression gets here.


@script.indicator(title="Int-typed length")
def main():
    length = 14 / 8  # int-typed on TradingView, 1.75 in Python
    return {
        "close": close,
        "highest_frac": ta.highest(length),
        "highest_int": ta.highest(1),
        "highest_src_frac": ta.highest(close, length),
        "highest_src_int": ta.highest(close, 1),
        "lowest_frac": ta.lowest(length),
        "lowest_int": ta.lowest(1),
        "sma_frac": ta.sma(close, length),
        "sma_int": ta.sma(close, 1),
        "cog_frac": ta.cog(close, 3 / 2),
        "cog_int": ta.cog(close, 1),
        "linreg_frac": ta.linreg(close, 6 / 4, 3 / 2),
        "linreg_int": ta.linreg(close, 1, 1),
        "valuewhen_frac": ta.valuewhen(close > 0, close, 3 / 2),
        "valuewhen_int": ta.valuewhen(close > 0, close, 1),
        "pivothigh_frac": ta.pivothigh(close, 4 / 2, 4 / 2),
        "pivothigh_int": ta.pivothigh(close, 2, 2),
        "pivotlow_frac": ta.pivotlow(close, 4 / 2, 4 / 2),
        "pivotlow_int": ta.pivotlow(close, 2, 2),
        "hist_frac": close[6 / 4],
        "hist_int": close[1],
    }


def __test_int_typed_length__(runner):
    """
    A fractional length that Pine's type system calls an int must behave exactly
    like the truncated int, in every function that consumes it as an integer.
    Before the truncation existed, the overloaded ones (``ta.highest``) raised
    "No matching implementation found" outright, while the plain ones were worse:
    ``ta.sma`` summed a 1-bar window and divided it by 1.75.
    """
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.na import NA

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    seed = 7717
    price = 100.0
    rows = []
    for bar in range(120):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        price += (seed / 0x7FFFFFFF - 0.5) * 4.0
        rows.append(OHLCV(timestamp=base_ts + bar * 1800, open=price, high=price + 1.5,
                          low=price - 1.5, close=price, volume=10.0))

    def is_na(value):
        # An na plot value arrives either as NA or as a bare nan, depending on
        # which side of the library produced it
        return isinstance(value, NA) or value != value

    pairs = ("highest", "highest_src", "lowest", "sma", "cog", "linreg",
             "valuewhen", "pivothigh", "pivotlow", "hist")
    compared = 0
    for i, (_candle, plot) in enumerate(runner(iter(rows)).run_iter()):
        for name in pairs:
            frac, exact = plot[f"{name}_frac"], plot[f"{name}_int"]
            if is_na(frac) or is_na(exact):
                assert is_na(frac) and is_na(exact), \
                    f"{name} na-disagrees at bar {i}: {frac} vs {exact}"
                continue
            assert frac == exact, f"{name} differs at bar {i}: {frac} vs {exact}"
            compared += 1

    assert compared > 500, f"too few non-na comparisons: {compared}"
