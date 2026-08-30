"""
@pyne

A length that grows back re-reads the whole window on the very bar it grows.

The extreme kept by ``ta.highest`` / ``ta.lowest`` only expires on AGE, so a value
evicted while the length was short stayed forgotten once the length reached back
over it again — the window answered from a stale, too-short reach for as many bars
as the growth spanned.

MEASURED on TradingView (wild-corpus indicator "Ichimoku Kinko hyo",
BINANCE:BTCUSDT 30m, 29075 bars): its Senkou length adapts down to 50 and climbs
back, and TradingView answers over the FULL requested window immediately at every
growth. Rescanning on growth reproduces all 29075 bars of both its ``ta.highest``
and ``ta.lowest``; the age-only rescan missed 24 and 154 of them. A SHRINKING
length needs no rescan — an extreme still inside the smaller window is the extreme
of that subset too.

The synthetic feed here isolates the shape: one deep extreme, a length that first
covers it, then shrinks past it, then grows back over it again.
"""
from pynecore.lib import script, ta, bar_index, low, high

# One deep low and one tall high at bar 5, flat noise everywhere else.
SPIKE_BAR = 5
LOWS = [100.0 + bar for bar in range(20)]
LOWS[SPIKE_BAR] = 10.0
HIGHS = [200.0 - bar for bar in range(20)]
HIGHS[SPIKE_BAR] = 900.0

WIDE = 10
NARROW = 3
SHRUNK_BARS = (10, 11)  # the spike is out of reach only here


def _length(bar: int) -> int:
    return NARROW if bar in SHRUNK_BARS else WIDE


@script.indicator(title="Growing length window")
def main():
    length = NARROW if bar_index in SHRUNK_BARS else WIDE
    return {
        "lowest": ta.lowest(low, length),
        "highest": ta.highest(high, length),
        "length": length,
    }


def _rows():
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    return [OHLCV(timestamp=base_ts + bar * 1800, open=lo + 5.0, high=hi, low=lo,
                  close=lo + 5.0, volume=10.0)
            for bar, (lo, hi) in enumerate(zip(LOWS, HIGHS))]


def __test_varying_length_window_is_always_the_plain_window__(runner):
    """ Every bar answers over exactly the length it asked for """
    for bar, (_candle, plot) in enumerate(runner(iter(_rows())).run_iter()):
        length = _length(bar)
        if bar < length - 1:
            assert plot["lowest"] != plot["lowest"], f"bar {bar}: expected na"
            continue
        window = slice(bar - length + 1, bar + 1)
        assert plot["lowest"] == min(LOWS[window]), f"bar {bar}: lowest {plot['lowest']}"
        assert plot["highest"] == max(HIGHS[window]), f"bar {bar}: highest {plot['highest']}"


def __test_the_spike_returns_the_bar_the_length_grows_back__(runner):
    """ The regression itself: bar 12 must see the spike again, not a bar later """
    growth_bar = max(SHRUNK_BARS) + 1
    assert growth_bar - WIDE + 1 <= SPIKE_BAR, "the wide window must reach the spike"
    values = [dict(plot) for _candle, plot in runner(iter(_rows())).run_iter()]
    assert values[growth_bar]["lowest"] == LOWS[SPIKE_BAR]
    assert values[growth_bar]["highest"] == HIGHS[SPIKE_BAR]


def __test_the_shrunk_window_really_loses_the_spike__(runner):
    """ Control: without the dip the test above would prove nothing """
    values = [dict(plot) for _candle, plot in runner(iter(_rows())).run_iter()]
    for bar in SHRUNK_BARS:
        assert values[bar]["lowest"] > LOWS[SPIKE_BAR], f"bar {bar} still sees the spike"
        assert values[bar]["highest"] < HIGHS[SPIKE_BAR], f"bar {bar} still sees the spike"
