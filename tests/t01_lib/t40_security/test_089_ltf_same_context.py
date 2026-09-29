"""
@pyne
"""
from pynecore.lib import array, bar_index, close, high, open, plot, request, script, syminfo, timeframe


@script.indicator(title="LTF Same Context Test", shorttitle="LSCT")
def main():
    # An LTF request at the chart's own symbol and timeframe is served inline
    # (no security child). TradingView still returns an ARRAY: one element per
    # bar, the chart bar itself — a tuple expression gives one-element columns.
    a = request.security_lower_tf(syminfo.tickerid, timeframe.period, close)
    h, bi = request.security_lower_tf(syminfo.tickerid, timeframe.period, [high, bar_index])
    up = request.security_lower_tf(syminfo.tickerid, timeframe.period, close > open)
    plot(array.size(a), "size_a")
    plot(array.get(a, 0) - close, "a0_minus_close")
    plot(array.size(h), "size_h")
    plot(array.get(h, 0) - high, "h0_minus_high")
    plot(array.get(bi, 0) - bar_index, "bi0_minus_bi")
    plot(array.size(up), "size_up")
    plot(1 if array.get(up, 0) == (close > open) else 0, "up0_eq")
    # Each bar gets a fresh array: popping it must not empty the next bar's.
    popped = array.pop(up)
    plot(array.size(up), "size_up_after_pop")
    plot(1 if popped == (close > open) else 0, "popped_eq")


def __test_ltf_same_context__(runner, log):
    """``request.security_lower_tf`` at the chart's own symbol and timeframe
    returns a fresh one-element array of the chart bar's value on every bar
    (measured on TradingView, CAPITALCOM:US500@30, 21165 bars)."""
    from pynecore.types.ohlcv import OHLCV

    ts0 = 1_735_689_600_000
    bars = [
        OHLCV(timestamp=ts0 + i * 300_000, open=100.0 + i, high=110.0 + 2 * i,
              low=90.0, close=100.0 + i + (1.0 if i % 2 else -1.0), volume=1.0)
        for i in range(6)
    ]
    rows = [dict(pv) for _candle, pv in runner(bars).run_iter()]
    assert len(rows) == len(bars)
    for i, pv in enumerate(rows):
        assert pv["size_a"] == 1, f"bar {i}: {pv}"
        assert pv["a0_minus_close"] == 0, f"bar {i}: {pv}"
        assert pv["size_h"] == 1, f"bar {i}: {pv}"
        assert pv["h0_minus_high"] == 0, f"bar {i}: {pv}"
        assert pv["bi0_minus_bi"] == 0, f"bar {i}: {pv}"
        assert pv["size_up"] == 1, f"bar {i}: {pv}"
        assert pv["up0_eq"] == 1, f"bar {i}: {pv}"
        assert pv["size_up_after_pop"] == 0, f"bar {i}: {pv}"
        assert pv["popped_eq"] == 1, f"bar {i}: {pv}"
    log.info("same-context LTF returns a one-element array per bar")
