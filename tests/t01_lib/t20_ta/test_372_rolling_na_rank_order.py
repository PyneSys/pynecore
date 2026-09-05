"""
@pyne

Rank order of the rolling window functions when the window holds na.

Measured on TradingView (BINANCE:BTCUSDT 30m, length 5, a source that is na for
its first three bars): an na does NOT sort to the end of the window the way it
does in the array face — it keeps a slot of its own, and a warm-up na keeps the
FRONT one. So the 100th percentile of ``[na, na, na, 3, 4]`` is 4, its 75th is
3.25 and its 0th is na — the exact opposite of the array face, where every one
of those is na except the 0th.

Two more laws come with it, measured on the same probes:

* an na BAR does not blank the answer — the window holds it like any other
  element (``percentile``) or refuses it and answers from what it already holds
  (``median``, ``mode``);
* the interpolation is evaluated unconditionally, so an na at the upper rank
  poisons the result even at a zero fraction.
"""
from pynecore.lib import script, bar_index, ta, na
from pynecore.types import Series

#: TradingView's answer per bar for the warm-up-na source, ``[p0, p50, p75, p100]``
__test_helper_LINEAR_WARMUP = {
    4: (None, None, 3.25, 4.0),
    5: (None, 3.0, 4.25, 5.0),
    6: (None, 4.0, 5.25, 6.0),
    7: (3.0, 5.0, 6.25, 7.0),
}

#: TradingView's rank readout for the same source: ranks 1..5 (20/40/60/80/100 %)
__test_helper_RANKS_WARMUP = {
    4: (None, None, None, 3.0, 4.0),
    5: (None, None, 3.0, 4.0, 5.0),
    6: (None, 3.0, 4.0, 5.0, 6.0),
    7: (3.0, 4.0, 5.0, 6.0, 7.0),
}

#: ``ta.median`` / ``ta.mode`` on the same source: the na bars never join the
#: window, so five non-na values first exist on bar 7
__test_helper_MEDIAN_MODE_WARMUP = {4: None, 5: None, 6: None, 7: (5.0, 3.0)}


@script.indicator(title="rolling na rank order")
def main():
    src: Series[float] = na if bar_index < 3 else float(bar_index)
    # An na arriving into a FILLED window takes the back slot instead, so the
    # low percentiles keep answering across it (measured: bar 7 of a source that
    # is na on every 7th bar answers 3 at 0 %, where the window is [3,4,5,6,na])
    scattered: Series[float] = na if bar_index % 7 == 0 else float(bar_index % 13)
    return {
        "l0": ta.percentile_linear_interpolation(src, 5, 0),
        "l50": ta.percentile_linear_interpolation(src, 5, 50),
        "l75": ta.percentile_linear_interpolation(src, 5, 75),
        "l100": ta.percentile_linear_interpolation(src, 5, 100),
        "r1": ta.percentile_nearest_rank(src, 5, 20),
        "r2": ta.percentile_nearest_rank(src, 5, 40),
        "r3": ta.percentile_nearest_rank(src, 5, 60),
        "r4": ta.percentile_nearest_rank(src, 5, 80),
        "r5": ta.percentile_nearest_rank(src, 5, 100),
        "median": ta.median(src, 5),
        "mode": ta.mode(src, 5),
        "s0": ta.percentile_linear_interpolation(scattered, 5, 0),
        "s50": ta.percentile_linear_interpolation(scattered, 5, 50),
        "s_median": ta.median(scattered, 5),
        "s_mode": ta.mode(scattered, 5),
    }


def __test_helper_check(bar: int, key: str, value, expected):
    if expected is None:
        # ``not (x == x)`` covers both na representations: an ``NA`` object
        # answers False to ``!=`` just as it does to ``==``
        assert not (value == value), f"bar {bar}: {key} must be na, got {value!r}"
    else:
        assert value == expected, f"bar {bar}: {key} expected {expected}, got {value!r}"


def __test_warmup_na_holds_the_front_ranks__(csv_reader, runner):
    """A warm-up na keeps the low ranks, so the high percentiles answer early"""
    with csv_reader('ma.csv', subdir="data") as cr:
        for i, (_, plot) in enumerate(runner(cr).run_iter()):
            if i in __test_helper_LINEAR_WARMUP:
                p0, p50, p75, p100 = __test_helper_LINEAR_WARMUP[i]
                __test_helper_check(i, "l0", plot["l0"], p0)
                __test_helper_check(i, "l50", plot["l50"], p50)
                __test_helper_check(i, "l75", plot["l75"], p75)
                __test_helper_check(i, "l100", plot["l100"], p100)
            if i in __test_helper_RANKS_WARMUP:
                for slot, expected in enumerate(__test_helper_RANKS_WARMUP[i], start=1):
                    __test_helper_check(i, f"r{slot}", plot[f"r{slot}"], expected)
            if i > 8:
                break


def __test_na_bar_does_not_blank_the_window__(csv_reader, runner):
    """An na bar joins the percentile window and is refused by median/mode"""
    with csv_reader('ma.csv', subdir="data") as cr:
        for i, (_, plot) in enumerate(runner(cr).run_iter()):
            if i in __test_helper_MEDIAN_MODE_WARMUP:
                expected = __test_helper_MEDIAN_MODE_WARMUP[i]
                __test_helper_check(i, "median", plot["median"], None if expected is None else expected[0])
                __test_helper_check(i, "mode", plot["mode"], None if expected is None else expected[1])
            if i == 7:
                # scattered source: bar 7 IS na, window [3, 4, 5, 6, na]
                __test_helper_check(i, "s0", plot["s0"], 3.0)
                __test_helper_check(i, "s50", plot["s50"], 5.0)
                # median/mode answer from the five non-na values they hold: [2..6]
                __test_helper_check(i, "s_median", plot["s_median"], 4.0)
                __test_helper_check(i, "s_mode", plot["s_mode"], 2.0)
            if i > 8:
                break
