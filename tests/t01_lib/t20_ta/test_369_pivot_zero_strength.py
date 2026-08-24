"""
@pyne

A pivot strength of zero is legal on either side.

TradingView rejects only a NEGATIVE strength (RE10001, "must be >= 0"); zero
simply means that side needs no confirmation. Measured on BINANCE:BTCUSDT 30m:
``ta.pivothigh(high, 0, 0)`` reports every bar's own high, ``(high, 5, 0)``
confirms the pivot on the bar that makes it -- no look-forward at all -- and
``(high, 0, 5)`` reports the bar five back once nothing beat it. The reference
covers all six combinations over 400 bars.
"""
from pynecore.lib import script, ta, plot, high, low


@script.indicator(title="Pivot Zero Strength", overlay=True)
def main():
    plot(ta.pivothigh(high, 5, 0), "ph_5_0")
    plot(ta.pivotlow(low, 5, 0), "pl_5_0")
    plot(ta.pivothigh(high, 0, 5), "ph_0_5")
    plot(ta.pivotlow(low, 0, 5), "pl_0_5")
    plot(ta.pivothigh(high, 0, 0), "ph_0_0")
    plot(ta.pivothigh(high, 5, 3), "ph_5_3")


# noinspection PyShadowingNames
def __test_pivot_zero_strength__(csv_reader, runner, dict_comparator):
    """ Zero-strength pivots follow the TradingView reference """
    with csv_reader('pivot_zero_strength.csv', subdir="data") as cr:
        bars = 0
        for candle, plot in runner(cr).run_iter():
            dict_comparator(plot, candle.extra_fields)
            bars += 1
        assert bars == 400, bars
