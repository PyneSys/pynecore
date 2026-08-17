"""
@pyne

ta.sar is na for the whole run once it is called conditionally.

Measured on TradingView (probe m571, BINANCE:BTCUSDT 30m, 28505 bars): a
``ta.sar(0.02, 0.02, 0.2)`` inside an ``if`` yields na on EVERY bar -- both when the
branch runs on half the bars and when it skips only every 100th one -- while a
``ta.atr(14)`` gated by the same branch keeps producing values there (14240 and
28206 non-na bars). The sar machine reads the previous bar's high and low from its
own window, which advances per CALL, so the first gated call finds nothing behind it
and the recurrence carries that na forward for good.

That is why ``sar`` must keep reading ``high[1]`` / ``low[1]`` and not the runner's
global ``lib._last_close``-style windows the way ``tr`` does (see
``test_365_conditional_tr_prev_close``): a global window would hand the machine real
prices across the gap and PyneCore would invent values where TradingView shows none.
The ``every`` and ``dense_atr`` plots are the controls -- without them the na
assertions would also hold on data that produces nothing at all.
"""
from pynecore.lib import script, ta, bar_index

# A rising trend with a couple of pullbacks, enough bars for both gates to bite
BARS = ((100.0, 90.0), (110.0, 100.0), (120.0, 85.0), (130.0, 112.0), (140.0, 120.0),
        (135.0, 118.0), (145.0, 125.0), (160.0, 140.0), (155.0, 130.0), (150.0, 128.0),
        (165.0, 142.0), (175.0, 155.0), (170.0, 150.0), (180.0, 160.0), (195.0, 172.0),
        (190.0, 168.0), (185.0, 160.0), (200.0, 175.0), (210.0, 190.0), (205.0, 185.0),
        (215.0, 195.0), (225.0, 205.0), (220.0, 200.0), (230.0, 210.0), (245.0, 222.0))


@script.indicator(title="Conditional sar")
def main():
    every = ta.sar(0.02, 0.02, 0.2)
    half = -1.0
    if bar_index % 2 == 0:
        half = ta.sar(0.02, 0.02, 0.2)
    dense = -1.0
    dense_atr = -1.0
    if bar_index % 10 != 0:
        dense = ta.sar(0.02, 0.02, 0.2)
        dense_atr = ta.atr(3)
    return {"every": every, "half": half, "dense": dense, "dense_atr": dense_atr}


def _rows():
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    return [OHLCV(timestamp=base_ts + bar * 1800, open=(h + lo) / 2, high=h, low=lo,
                  close=(h + lo) / 2, volume=10.0)
            for bar, (h, lo) in enumerate(BARS)]


def __test_every_bar_sar_produces_values__(runner):
    """ Control: called on every bar, sar leaves na right after the first bar """
    for bar, (_candle, plot) in enumerate(runner(iter(_rows())).run_iter()):
        value = plot["every"]
        assert (value != value) == (bar == 0), f"bar {bar}: unexpected na state {value}"


def __test_half_gated_sar_is_na_everywhere__(runner):
    """ Called on even bars only, sar never gets a previous bar of its own """
    for bar, (_candle, plot) in enumerate(runner(iter(_rows())).run_iter()):
        value = plot["half"]
        if bar % 2:
            assert value == -1.0, f"bar {bar}: the branch must not run"
            continue
        assert value != value, f"bar {bar}: gated sar should be na, got {value}"


def __test_sparsely_gated_sar_is_na_everywhere__(runner):
    """ One skipped bar is enough: sar stays na, while the gated atr does not """
    seen_atr = 0
    for bar, (_candle, plot) in enumerate(runner(iter(_rows())).run_iter()):
        if bar % 10 == 0:
            assert plot["dense"] == -1.0, f"bar {bar}: the branch must not run"
            continue
        value = plot["dense"]
        assert value != value, f"bar {bar}: gated sar should be na, got {value}"
        atr_value = plot["dense_atr"]
        if atr_value == atr_value:
            seen_atr += 1
    assert seen_atr > 0, "the gated atr must still produce values, or this proves nothing"
