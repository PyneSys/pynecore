"""
@pyne

``ta.highest`` / ``ta.lowest`` / ``ta.highestbars`` / ``ta.lowestbars`` run on two
machines, chosen by the qualifier of ``length``.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, probes ``qual`` and ``late``, 30657
bars, every value matched; the reference here is their first 450 bars):

- a const / input / simple length runs the ``length + 1`` ring addressed by the chart
  bar. A never-written slot holds 0.0, so a call skipping most bars returns 0.0 while
  its window still reaches one (``c_lo_g2``); the rescan stops at a slot an na call
  wrote (``c_hna``); a site whose first call is na still takes its length, so the
  next call does not rescan as a growth from 0 (``lo_na5``);
- a ``series`` length forward-fills the bars a conditional call skips (``s_*``
  columns), even when its value never changes: ``ser8`` is 8 on every bar, but it
  reads ``bar_index``, so it is series.
"""
from pynecore.lib import script, ta, high, low, bar_index, na, plot


@script.indicator(title="Window extreme machines")
def main():
    gate1 = bar_index % 50 < 17
    gate2 = bar_index % 37 < 5
    alt = bar_index % 2 == 0
    hna = na if bar_index % 23 == 0 else high
    ser8 = 8 if bar_index >= 0 else 99
    ser10 = 10 if bar_index >= 0 else 99
    ser7 = 7 if bar_index >= 0 else 99

    c_lo_g1 = na
    s_lo_g1 = na
    if gate1:
        c_lo_g1 = ta.lowest(low, 8)
        s_lo_g1 = ta.lowest(low, ser8)
    plot(c_lo_g1, "c_lo_g1")
    plot(s_lo_g1, "s_lo_g1")

    c_lo_g2 = na
    if gate2:
        c_lo_g2 = ta.lowest(low, 6)
    plot(c_lo_g2, "c_lo_g2")

    s_hi_alt = na
    c_hb_alt = na
    s_hb_alt = na
    if alt:
        s_hi_alt = ta.highest(high, ser10)
        c_hb_alt = ta.highestbars(high, 10)
        s_hb_alt = ta.highestbars(high, ser10)
    plot(s_hi_alt, "s_hi_alt")
    plot(c_hb_alt, "c_hb_alt")
    plot(s_hb_alt, "s_hb_alt")

    c_hna = na
    s_hna = na
    if bar_index % 3 != 1:
        c_hna = ta.highest(hna, 7)
        s_hna = ta.highest(hna, ser7)
    plot(c_hna, "c_hna")
    plot(s_hna, "s_hna")

    lob_late = na
    if bar_index >= 100 and bar_index % 3 != 2:
        lob_late = ta.lowestbars(low, 9)
    plot(lob_late, "lob_late")

    lo_na5 = na
    if bar_index % 5 == 0:
        lo_na5 = ta.lowest(na if bar_index % 7 == 0 else low, 6)
    plot(lo_na5, "lo_na5")


# noinspection PyShadowingNames
def __test_window_extreme_machines__(csv_reader, runner, dict_comparator, log):
    """ Ring refinements and the series machine, bit-exact """
    with csv_reader('window_extreme_machines.csv', subdir="data") as cr:
        for candle, plot in runner(cr).run_iter():
            dict_comparator(plot, candle.extra_fields, abs_tol=0.0, rel_tol=0.0)
