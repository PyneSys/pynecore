"""HTFAggregator unit tests — chart-side developing HTF bar accumulation."""
from zoneinfo import ZoneInfo

from pynecore.core.htf_aggregator import HTFAggregator


_UTC = ZoneInfo("UTC")


def _ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Build a UTC epoch ms timestamp from calendar fields."""
    from datetime import datetime
    return int(datetime(year, month, day, hour, minute, tzinfo=_UTC).timestamp() * 1000)


def __test_first_chart_bar_starts_developing__(log):
    agg = HTFAggregator("60", _UTC)
    ts = _ms(2026, 5, 21, 10, 30)
    is_new, dev, closed = agg.update(
        ts, chart_open=1.0, chart_high=1.2, chart_low=0.9, chart_close=1.1,
        chart_volume=10.0,
    )
    assert is_new is True
    assert closed is None
    assert dev.period_start == _ms(2026, 5, 21, 10, 0)
    assert (dev.open, dev.high, dev.low, dev.close, dev.volume) == (
        1.0, 1.2, 0.9, 1.1, 10.0,
    )


def __test_accumulates_within_same_period__(log):
    agg = HTFAggregator("60", _UTC)
    base = _ms(2026, 5, 21, 10, 0)
    agg.update(base + 0,           1.0, 1.2, 0.9, 1.1, 10.0)
    agg.update(base + 5 * 60_000,  1.1, 1.3, 1.0, 1.05, 5.0)
    is_new, dev, closed = agg.update(
        base + 10 * 60_000, 1.05, 1.4, 0.95, 1.2, 3.0,
    )
    assert is_new is False
    assert closed is None
    # open stays the FIRST chart bar's open
    assert dev.open == 1.0
    assert dev.high == 1.4   # max(1.2, 1.3, 1.4)
    assert dev.low == 0.9    # min(0.9, 1.0, 0.95)
    assert dev.close == 1.2  # latest chart close
    assert dev.volume == 18.0  # 10 + 5 + 3


def __test_period_boundary_emits_closed_then_starts_fresh__(log):
    agg = HTFAggregator("60", _UTC)
    # Hour 10
    agg.update(_ms(2026, 5, 21, 10, 0),  1.0, 1.2, 0.9, 1.1, 10.0)
    agg.update(_ms(2026, 5, 21, 10, 30), 1.1, 1.5, 1.0, 1.3, 5.0)
    # Hour 11 opens — previous period closes
    is_new, dev, closed = agg.update(
        _ms(2026, 5, 21, 11, 0), 1.3, 1.4, 1.25, 1.35, 4.0,
    )
    assert is_new is True
    assert closed is not None
    assert closed.period_start == _ms(2026, 5, 21, 10, 0)
    assert (closed.open, closed.high, closed.low, closed.close) == (
        1.0, 1.5, 0.9, 1.3,
    )
    assert closed.volume == 15.0
    # Fresh developing for hour 11
    assert dev.period_start == _ms(2026, 5, 21, 11, 0)
    assert (dev.open, dev.high, dev.low, dev.close, dev.volume) == (
        1.3, 1.4, 1.25, 1.35, 4.0,
    )


def __test_current_property_returns_state__(log):
    agg = HTFAggregator("60", _UTC)
    assert agg.current is None
    agg.update(_ms(2026, 5, 21, 10, 0), 1.0, 1.0, 1.0, 1.0, 1.0)
    assert agg.current is not None
    assert agg.current.period_start == _ms(2026, 5, 21, 10, 0)
    agg.reset()
    assert agg.current is None


def __test_daily_timeframe_uses_tz_for_boundary__(log):
    # NY tz: a 2026-05-21 09:30 NY bar belongs to 2026-05-21 00:00 NY day.
    ny = ZoneInfo("America/New_York")
    agg = HTFAggregator("1D", ny)
    from datetime import datetime
    chart_ts = int(datetime(2026, 5, 21, 9, 30, tzinfo=ny).timestamp() * 1000)
    is_new, dev, closed = agg.update(chart_ts, 1.0, 1.0, 1.0, 1.0, 1.0)
    expected_period_start = int(
        datetime(2026, 5, 21, 0, 0, tzinfo=ny).timestamp() * 1000
    )
    assert is_new is True
    assert closed is None
    assert dev.period_start == expected_period_start


def __test_intra_bar_updates_do_not_inflate_volume__(log):
    """Live providers (CCXT et al.) emit repeated updates for the same chart
    bar carrying the running cumulative candle volume. The aggregator must
    deduplicate these so the HTF developing/closed volume reflects one
    contribution per chart bar, not one per tick.
    """
    agg = HTFAggregator("60", _UTC)
    base = _ms(2026, 5, 21, 10, 0)
    # First chart bar — three intra-bar ticks with running cumulative volume.
    agg.update(base + 0,           1.0, 1.0, 1.0, 1.0, 3.0)
    agg.update(base + 0,           1.0, 1.1, 1.0, 1.05, 7.0)
    _, dev, _ = agg.update(base + 0, 1.0, 1.2, 0.95, 1.15, 10.0)
    # All ticks share the same chart bar — running cumulative ends at 10.
    assert dev.volume == 10.0
    assert dev.high == 1.2
    assert dev.low == 0.95
    assert dev.close == 1.15

    # Second chart bar — also cumulative within its own ticks.
    agg.update(base + 5 * 60_000, 1.15, 1.15, 1.15, 1.15, 2.0)
    _, dev, _ = agg.update(base + 5 * 60_000, 1.15, 1.3, 1.1, 1.25, 6.0)
    # HTF developing volume = first bar (10) + second bar (6).
    assert dev.volume == 16.0
    assert dev.high == 1.3
    assert dev.low == 0.95
    assert dev.close == 1.25
