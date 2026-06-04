"""
@pyne
"""
from pynecore.lib import close, open, plot, script
from pynecore.core.series import inline_series


@script.indicator("Inline Series Isolation", shorttitle="ISI")
def main():
    # Two DISTINCT inline_series call sites in the same scope. Each must keep its
    # own buffer: a == close[1], b == open[1]. Issue #61: before the fix every
    # inline_series call site shared one module-global buffer, so b's same-bar
    # write overwrote a's slot and a returned open[1] instead of close[1].
    a = inline_series(close, 1)
    b = inline_series(open, 1)
    plot(a, "a")
    plot(b, "b")
    plot(close, "c")
    plot(open, "o")


def __test_inline_series_isolation__(csv_reader, runner, log):
    """Two distinct inline_series() call sites produce independent shifted series."""
    from pynecore.types.na import NA

    prev_close = None
    prev_open = None
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, pv) in enumerate(runner(cr).run_iter()):
            a = pv.get('a')
            b = pv.get('b')
            if i == 0:
                assert isinstance(a, NA), f"bar 0: a should be na, got {a!r}"
                assert isinstance(b, NA), f"bar 0: b should be na, got {b!r}"
            else:
                assert a == prev_close, \
                    f"bar {i}: inline_series(close,1)={a} != close[1]={prev_close}"
                assert b == prev_open, \
                    f"bar {i}: inline_series(open,1)={b} != open[1]={prev_open}"
            prev_close = pv.get('c')
            prev_open = pv.get('o')

    log.info("inline_series isolation: distinct call sites stay independent")
