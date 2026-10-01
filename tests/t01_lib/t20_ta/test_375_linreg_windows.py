"""
@pyne

``ta.linreg`` runs on two machines, chosen by the qualifier of ``length``.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, probes ``lrna``/``lrnb``, 30657 bars,
every value bit-exact; the reference here is their first 450 bars):

- a const / input / simple length runs the ``ta.highest`` ring: ``length + 1`` slots
  addressed by the chart bar, written only on the bars the call runs, a
  never-written slot holding 0.0 -- so a conditional call regresses over stale
  slots, and its first calls over zeros;
- a ``series`` length (the type pass derives it, e.g. ``ls8`` below) forward-fills
  the bars a conditional call skips, a bar before the first call is na, and a
  growing length reads the true history;
- on both, an na is stored like any value and a window holding one answers na, so a
  single na blanks the next ``length`` bars.
"""
from pynecore.lib import script, ta, close, bar_index, na, plot


@script.indicator(title="Linreg windows")
def main():
    s1 = na if bar_index % 50 == 7 else close
    s2 = na if bar_index % 200 < 30 and bar_index > 100 else close
    ls8 = 8 if bar_index % 40 < 20 else 11
    lsg = 5 if bar_index < 300 else 20

    plot(ta.linreg(s1, 10, 0), "l1")
    plot(ta.linreg(s2, 15, 2), "l2")
    l6 = na
    if bar_index % 3 != 0:
        l6 = ta.linreg(close, 8, 0)
    plot(l6, "l6")

    m2 = na
    if bar_index % 3 != 0:
        m2 = ta.linreg(close, ls8, 0)
    plot(m2, "m2")
    m3 = na
    if bar_index % 3 != 0:
        m3 = ta.linreg(s1, ls8, 1)
    plot(m3, "m3")
    plot(ta.linreg(close, lsg, 0), "m4")
    m5 = na
    if bar_index % 5 < 2:
        m5 = ta.linreg(close, 4, 1)
    plot(m5, "m5")


# noinspection PyShadowingNames
def __test_linreg_windows__(csv_reader, runner, dict_comparator, log):
    """ Ring and series machines, na windows and conditional calls, bit-exact """
    with csv_reader('linreg_windows.csv', subdir="data") as cr:
        for candle, plot in runner(cr).run_iter():
            dict_comparator(plot, candle.extra_fields, abs_tol=0.0, rel_tol=0.0)
