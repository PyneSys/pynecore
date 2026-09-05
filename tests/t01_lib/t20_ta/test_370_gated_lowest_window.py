"""
@pyne

A gated ``ta.lowest`` serves the stale slot of its own per-call-site window.

MEASURED on TradingView (wild-corpus script "Twin Range Filter Algo",
BINANCE:BTCUSDT 30m, 28827 bars): a ``ta.lowest(low, 10)`` sitting inside a
position-state ``if`` does NOT return the lowest of the last 10 chart bars. Every
call site owns a window buffer of ``length + 1`` slots addressed by the CHART BAR
(``bar_index % (length + 1)``) and written only on the bars the call runs, so a
skipped bar leaves its slot holding the value written a whole capacity earlier and
the extreme is taken over that stale value. Reading the source's own bar history
instead misses 872 of those 28827 values; with this addressing all of them match.

The synthetic feed here reproduces the same shape: the call is skipped for a run of
bars longer than the window, and the first calls after the gap must serve values
from before it -- values a plain 10-bar window could never return.
"""
from pynecore.lib import script, ta, bar_index, low

# A long descent, a gap-spanning plateau, then a climb: the pre-gap lows are far
# below everything the post-gap window can see on its own.
__test_helper_LOWS = [100.0 - bar for bar in range(12)] + [200.0 + bar for bar in range(24)]

# The call is skipped on these bars -- a run of 12, longer than the window itself.
SKIPPED = set(range(12, 24))

LENGTH = 4
__test_helper_CAPACITY = LENGTH + 1


@script.indicator(title="Gated lowest")
def main():
    gated = -1.0
    if bar_index not in SKIPPED:
        gated = ta.lowest(low, LENGTH)
    return {"gated": gated, "free": ta.lowest(low, LENGTH)}


def __test_helper_rows():
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    return [OHLCV(timestamp=base_ts + bar * 1800, open=lo + 5.0, high=lo + 10.0, low=lo,
                  close=lo + 5.0, volume=10.0)
            for bar, lo in enumerate(__test_helper_LOWS)]


def __test_helper_expected_gated() -> list[float]:
    """The measured machine, spelled out independently of the implementation."""
    window = [float("nan")] * __test_helper_CAPACITY
    out: list[float] = []
    for bar, lo in enumerate(__test_helper_LOWS):
        if bar in SKIPPED:
            out.append(-1.0)
            continue
        window[bar % __test_helper_CAPACITY] = lo
        if bar < LENGTH - 1:
            out.append(float("nan"))
            continue
        # ``avail`` counts bars since the site's first call, capped at the window
        seen = min(bar + 1, LENGTH)
        out.append(min(window[(bar - i) % __test_helper_CAPACITY] for i in range(seen)))
    return out


def __test_gated_lowest_reads_its_own_stale_window__(runner):
    """ After the gap the window still holds the pre-gap lows, one capacity apart """
    expected = __test_helper_expected_gated()
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        value = plot["gated"]
        want = expected[bar]
        if want != want:
            assert value != value, f"bar {bar}: expected na, got {value}"
            continue
        assert value == want, f"bar {bar}: expected {want}, got {value}"


def __test_gap_serves_values_no_plain_window_could__(runner):
    """ The stale slots are the whole point: the ungated call never sees them """
    differed = 0
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        if bar in SKIPPED:
            continue
        gated, free = plot["gated"], plot["free"]
        if gated == gated and free == free and gated != free:
            differed += 1
            assert gated < free, f"bar {bar}: a stale slot can only be lower here"
    assert differed >= LENGTH - 1, f"the gap must colour at least the next window ({differed})"


def __test_ungated_lowest_is_the_plain_window__(runner):
    """ Control: with no gate the window is exactly the last ``length`` bars """
    for bar, (_candle, plot) in enumerate(runner(iter(__test_helper_rows())).run_iter()):
        value = plot["free"]
        if bar < LENGTH - 1:
            assert value != value, f"bar {bar}: expected na, got {value}"
            continue
        assert value == min(__test_helper_LOWS[bar - LENGTH + 1:bar + 1]), f"bar {bar}: {value}"
