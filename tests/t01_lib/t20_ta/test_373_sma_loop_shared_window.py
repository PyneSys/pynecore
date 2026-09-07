"""
@pyne

A ``ta.sma`` call site inside a ``for`` loop whose length moves per iteration.

One call site serves every iteration of the bar, so a length that changes
between them walks the SHARED window from the length the previous bar left
behind to its own — evicting when it shrinks, admitting when it grows. That
walk is the machine's hot path, and it is memoized across the iterations of a
bar; the reference here is what the walk has to keep landing on.

MEASURED on TradingView (FX:EURUSD 60m), the reference beside this script. The
three loops cover the three shapes the walk takes: a length that only grows per
iteration (every iteration evicts down from the previous bar's largest), one
that only shrinks (every iteration admits up from the smallest), and one that
jumps around. Note the reference cannot separate the shared window from a
per-iteration one — the two models differ far below the comparison tolerance —
what it pins is that the walk and its memo still produce the right sums.
"""
from pynecore.lib import script, ta, close, hl2, plot
from pynecore.types.na import NA


@script.indicator(title="sma loop shared window", shorttitle="smaloop")
def main():
    asc = 0.0
    p1 = 5
    for _i in range(10):
        asc += ta.sma(close, p1)
        p1 += 5
    desc = 0.0
    p2 = 50
    for _j in range(10):
        desc += ta.sma(close, p2)
        p2 -= 5
    mix = 0.0
    p3 = 7
    for _k in range(6):
        mix += ta.sma(close, p3)
        p3 = 3 + (p3 * 5) % 41
    plot(hl2, "hl2")
    plot(asc, "asc")
    plot(desc, "desc")
    plot(mix, "mix")


# noinspection PyShadowingNames
def __test_sma_loop_shared_window__(csv_reader, runner, dict_comparator):
    """ ta.sma in a loop with a per-iteration length """
    with csv_reader('sma_loop_shared_window.csv', subdir="data") as cr:
        for candle, plot in runner(cr).run_iter():
            # TradingView writes its na as 1e+100 in this export
            expected = {key: NA(float) if value == 1e+100 else value
                        for key, value in candle.extra_fields.items()}
            dict_comparator(plot, expected)
