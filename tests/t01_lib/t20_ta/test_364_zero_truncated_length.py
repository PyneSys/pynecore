"""
@pyne

A length that truncates to zero is an error, not a quietly empty window.

``median``, ``mode`` and the two percentile machines are the only rolling ``ta``
functions that accept an na length (see test_362); a length of exactly 0 is an
error in this family too, exactly as everywhere else. An int-typed Pine value can
still carry a fraction, so ``4 / 8`` reaches them as 0.5 and truncates to that
invalid 0 -- which means the domain check has to run on the truncated value.
Validating first let 0.5 pass ``> 0``, and the functions then ran with an empty
window: ``median`` indexed an empty heap and crashed, while the other three
answered na forever instead of reporting the bad argument.
"""
from pynecore.lib import script, close, ta


@script.indicator(title="Zero-truncated length")
def main():
    zero = 4 / 8  # int-typed on TradingView, 0.5 in Python -> an invalid 0
    caught = 0
    try:
        ta.median(close, zero)
    except AssertionError:
        caught += 1
    try:
        ta.mode(close, zero)
    except AssertionError:
        caught += 1
    try:
        ta.percentile_nearest_rank(close, zero, 100)
    except AssertionError:
        caught += 1
    try:
        ta.percentile_linear_interpolation(close, zero, 50)
    except AssertionError:
        caught += 1
    return {"caught": caught}


def __test_zero_truncated_length__(runner):
    """Every function of the na-length family rejects a length that truncates to 0"""
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    rows = [OHLCV(timestamp=base_ts + bar * 1800, open=1.0 + bar, high=1.0 + bar,
                  low=1.0 + bar, close=1.0 + bar, volume=10.0) for bar in range(4)]

    bars = 0
    for i, (_candle, plot) in enumerate(runner(iter(rows)).run_iter()):
        assert plot["caught"] == 4, f"bar {i}: only {plot['caught']} of 4 rejected the length"
        bars += 1
    assert bars == 4
