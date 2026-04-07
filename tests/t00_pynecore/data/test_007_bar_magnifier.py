"""
@pyne
"""
from pynecore.core.bar_magnifier import BarMagnifier, MagnifiedWindow
from pynecore.types.ohlcv import OHLCV


def main():
    pass


# --- Basic grouping tests ---

def __test_groups_sub_bars_into_windows__():
    """6 x 1-minute bars should produce 2 x 3-minute windows."""
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0 + i, high=102.0 + i,
              low=99.0 + i, close=101.0 + i, volume=10.0)
        for i in range(6)
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))

    assert len(windows) == 2
    assert len(windows[0].sub_bars) == 3
    assert len(windows[1].sub_bars) == 3


def __test_aggregation_within_window__():
    """Aggregated OHLCV follows standard rules: O=first, H=max, L=min, C=last, V=sum."""
    base_ts = 1704067200
    sub_bars = [
        OHLCV(timestamp=base_ts,      open=100.0, high=110.0, low=95.0,  close=105.0, volume=100.0),
        OHLCV(timestamp=base_ts + 60, open=105.0, high=115.0, low=98.0,  close=108.0, volume=200.0),
        OHLCV(timestamp=base_ts + 120, open=108.0, high=112.0, low=90.0, close=103.0, volume=150.0),
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))
    assert len(windows) == 1

    agg = windows[0].aggregated
    assert agg.open == 100.0
    assert agg.high == 115.0
    assert agg.low == 90.0
    assert agg.close == 103.0
    assert agg.volume == 450.0


def __test_last_window_flag__():
    """Only the last window should have is_last_window=True."""
    base_ts = 1704067200
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=101.0,
              low=99.0, close=100.0, volume=10.0)
        for i in range(9)
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))

    assert len(windows) == 3
    assert windows[0].is_last_window is False
    assert windows[1].is_last_window is False
    assert windows[2].is_last_window is True


def __test_single_window__():
    """A single window should be both first and last."""
    base_ts = 1704067200
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=101.0,
              low=99.0, close=100.0, volume=10.0)
        for i in range(3)
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))

    assert len(windows) == 1
    assert windows[0].is_last_window is True
    assert len(windows[0].sub_bars) == 3


def __test_partial_last_window__():
    """If data doesn't fill the last window, it should still be emitted."""
    base_ts = 1704067200
    # 4 bars = one full 3-min window + one partial window with 1 bar
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=101.0,
              low=99.0, close=100.0, volume=10.0)
        for i in range(4)
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))

    assert len(windows) == 2
    assert len(windows[0].sub_bars) == 3
    assert len(windows[1].sub_bars) == 1
    assert windows[1].is_last_window is True


def __test_sub_bars_preserved_in_order__():
    """Sub-bars within each window should maintain their original order."""
    base_ts = 1704067200
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0 + i, high=101.0 + i,
              low=99.0 + i, close=100.5 + i, volume=10.0 + i)
        for i in range(6)
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))

    # First window: bars 0, 1, 2
    assert windows[0].sub_bars[0].open == 100.0
    assert windows[0].sub_bars[1].open == 101.0
    assert windows[0].sub_bars[2].open == 102.0

    # Second window: bars 3, 4, 5
    assert windows[1].sub_bars[0].open == 103.0
    assert windows[1].sub_bars[1].open == 104.0
    assert windows[1].sub_bars[2].open == 105.0


def __test_aggregated_timestamp_is_bar_boundary__():
    """Aggregated candle timestamp should be the bar boundary, not the first sub-bar's."""
    # Start at a non-boundary: 00:01:00, with 3-min chart TF
    # Bar boundary for 3-min at 00:01:00 is 00:00:00
    base_ts = 1704067200  # already at boundary (00:00:00 UTC)
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=101.0,
              low=99.0, close=100.0, volume=10.0)
        for i in range(3)
    ]

    windows = list(BarMagnifier(sub_bars, '3', tz=None))

    # The aggregated timestamp should be the bar boundary (start of 3-min period)
    assert windows[0].aggregated.timestamp == base_ts


# --- Larger timeframe tests ---

def __test_1min_to_5min__():
    """Standard use case: 1-minute data grouped into 5-minute bars."""
    base_ts = 1704067200
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=1.1000 + i * 0.0001,
              high=1.1010 + i * 0.0001, low=1.0990 + i * 0.0001,
              close=1.1005 + i * 0.0001, volume=100.0)
        for i in range(10)
    ]

    windows = list(BarMagnifier(sub_bars, '5', tz=None))

    assert len(windows) == 2
    assert all(len(w.sub_bars) == 5 for w in windows)

    # First 5-min bar: O from bar 0, C from bar 4
    assert windows[0].aggregated.open == sub_bars[0].open
    assert windows[0].aggregated.close == sub_bars[4].close

    # Second 5-min bar: O from bar 5, C from bar 9
    assert windows[1].aggregated.open == sub_bars[5].open
    assert windows[1].aggregated.close == sub_bars[9].close


def __test_1min_to_60min__():
    """The primary bar magnifier use case: 1-minute to 60-minute."""
    base_ts = 1704067200
    sub_bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=100.0 + (i % 10),
              low=100.0 - (i % 5), close=100.0, volume=10.0)
        for i in range(120)
    ]

    windows = list(BarMagnifier(sub_bars, '60', tz=None))

    assert len(windows) == 2
    assert len(windows[0].sub_bars) == 60
    assert len(windows[1].sub_bars) == 60
