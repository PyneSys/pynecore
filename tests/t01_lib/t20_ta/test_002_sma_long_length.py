"""
@pyne
"""
from pynecore.lib import script, close, ta


@script.indicator(title="SMA Long Length", shorttitle="sma_long")
def main():
    return {
        "close": close,
        "sma3": ta.sma(close, 3),
        "sma500": ta.sma(close, 500),
        "sma1000": ta.sma(close, 1000),
    }


def __test_sma_long_length__(runner):
    """
    ``ta.sma`` with a length beyond the default 500-bar ``max_bars_back`` must keep
    producing values. The sliding sum drops the value leaving the window via
    ``src[length]``; when ``length`` exceeded the per-series buffer that read
    returned na and poisoned the running sum, collapsing the moving average to na
    right after warmup (only a single bar ever matched). Length 500 (buffer boundary)
    stayed correct, length 1000 did not.
    """
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.na import NA

    n_bars = 1300
    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    step = 1800  # 30 minutes

    # Deterministic, varying, non-degenerate close series (LCG pseudo-random walk)
    # so the sliding-window removal is genuinely exercised numerically.
    gen_closes: list[float] = []
    seed = 12345
    price = 100.0
    for _ in range(n_bars):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        price += (seed / 0x7FFFFFFF - 0.5) * 2.0
        gen_closes.append(price)

    def ohlcv_iter():
        for bi, c in enumerate(gen_closes):
            yield OHLCV(timestamp=base_ts + bi * step, open=c, high=c + 1.0, low=c - 1.0,
                        close=c, volume=10.0)

    # Compare against the ``close`` values the script actually saw (the engine may
    # quantize prices), so this pins the sliding-sum correctness, not price
    # representation.
    seen: list[float] = []

    def rolling_mean(idx: int, length: int) -> float:
        window = seen[idx - length + 1: idx + 1]
        return sum(window) / length

    for i, (_candle, plot) in enumerate(runner(ohlcv_iter()).run_iter()):
        seen.append(plot["close"])
        # sma1000: na during warmup, a value once warm — the regression (pre-fix it
        # stayed na for the whole run after a single bar).
        assert (plot["sma1000"] is None or isinstance(plot["sma1000"], NA)) == (i < 999), \
            f"sma1000 warmup/na boundary wrong at bar {i}"
        if i >= 999:
            assert abs(plot["sma1000"] - rolling_mean(i, 1000)) < 1e-6, \
                f"sma1000 value wrong at bar {i}"
        # sma500 (buffer boundary) and sma3 must also stay correct
        if i >= 499:
            assert abs(plot["sma500"] - rolling_mean(i, 500)) < 1e-6, \
                f"sma500 value wrong at bar {i}"
        if i >= 2:
            assert abs(plot["sma3"] - rolling_mean(i, 3)) < 1e-6, \
                f"sma3 value wrong at bar {i}"
