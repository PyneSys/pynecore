"""
@pyne
"""
from pynecore.lib import close, plot, script, na
from pynecore.types import Series


def _pair(x):
    return x, x * 2.0


@script.indicator("Tuple Unpack Series History", shorttitle="TUSH")
def main():
    # A Series that receives its value through tuple unpacking (``a, b = f()``)
    # must record per-bar history exactly like a directly-assigned Series, so
    # that ``a[1]`` yields the previous bar's value. PyneComp emits such series
    # as a bare ``a: Series`` declaration (no initializer, hence no per-bar
    # ``add()``) followed by the tuple assignment; before the fix the unpack only
    # rebound the local name and never advanced the buffer, so ``a[1]`` was
    # always na. This is the exact pattern behind the wild "Adaptive Trend Flow"
    # strategy, whose ``[trend, level] = get_trend_state(...)`` made ``trend[1]``
    # permanently na and suppressed every entry.
    a: Series
    a, b = _pair(close)
    c: Series = close  # control: direct assignment, whose history already works
    plot(a, "a_cur")
    plot(c, "c_cur")
    plot(-1.0 if na(a[1]) else a[1], "a_prev")
    plot(-1.0 if na(c[1]) else c[1], "c_prev")


def __test_tuple_unpack_series_history__(csv_reader, runner, log):
    """A tuple-unpacked Series keeps history identical to a direct assignment."""
    bars = 0
    saw_history = False
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, pv) in enumerate(runner(cr).run_iter()):
            assert pv['a_cur'] == pv['c_cur'], \
                f"bar {i}: current value diverged a={pv['a_cur']} c={pv['c_cur']}"
            assert pv['a_prev'] == pv['c_prev'], \
                f"bar {i}: history a[1]={pv['a_prev']} != c[1]={pv['c_prev']}"
            if i >= 1:
                # Past the first bar the previous value must be a real number,
                # proving the tuple-unpacked buffer actually advanced.
                assert pv['a_prev'] != -1.0, f"bar {i}: tuple-unpacked a[1] stayed na"
                saw_history = True
            bars += 1

    assert saw_history and bars > 10, f"expected >10 bars with history, got {bars}"
    log.info("tuple-unpacked series history matches direct assignment over %d bars", bars)
