"""
@pyne

``ta.valuewhen`` remembers the source value of the last matching bar, so a na
source on a bar the condition is FALSE must not blank the remembered value --
which an argument-level na guard used to do. A na source ON a matching bar is a
different thing: it is recorded verbatim, still consumes an occurrence, and the
older occurrences stay reachable past it.

The reference was measured on TradingView (CAPITALCOM:EURUSD 30m).
"""
from pynecore.lib import script, ta, close, bar_index, na, plot


@script.indicator(title="Valuewhen NA Test", shorttitle="valuewhen_na")
def main():
    cond = bar_index % 3 == 0
    src = na if bar_index % 2 == 0 else close
    plot(ta.valuewhen(cond, src, 0), "v0")
    plot(ta.valuewhen(cond, src, 1), "v1")
    plot(ta.valuewhen(cond, src, 2), "v2")
    plot(ta.valuewhen(cond, close, 0), "w0")


# noinspection PyShadowingNames
def __test_valuewhen_na__(csv_reader, runner, dict_comparator, log):
    """ Valuewhen with na sources """
    with csv_reader('valuewhen_na.csv', subdir="data") as cr:
        for candle, plot in runner(cr).run_iter():
            dict_comparator(plot, candle.extra_fields)
