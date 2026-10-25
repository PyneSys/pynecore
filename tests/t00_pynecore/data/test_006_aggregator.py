"""
@pyne
"""
import pytest
import tempfile
import time
from pathlib import Path

from pynecore.core.aggregator import validate_aggregation, aggregate_ohlcv, _merge_candles
from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.ohlcv import OHLCVReader
from pynecore.types.ohlcv import OHLCV


def main():
    pass


def _create_ohlcv_file(path: Path, candles: list[OHLCV], period: str) -> None:
    """Helper to write candles into an .ohlcv file with its declared period."""
    with OHLCVWriter(path, period, truncate=True) as writer:
        for c in candles:
            writer.write(c)


def _read_all(path: Path) -> list[OHLCV]:
    """Helper to read all candles from an .ohlcv file."""
    result = []
    with OHLCVReader(path) as reader:
        for c in reader.read_from(reader.start_timestamp):
            result.append(c)
    return result


# --- Validation tests ---

def __test_validate_aggregation_ok__():
    """Valid aggregation pairs should not raise."""
    validate_aggregation("5", "60")      # 5min → 1hour
    validate_aggregation("1D", "1W")     # daily → weekly
    validate_aggregation("60", "1D")     # 1hour → daily
    validate_aggregation("15", "60")     # 15min → 1hour


def __test_validate_aggregation_downscale_fails__():
    """Downscaling must raise ValueError."""
    with pytest.raises(ValueError, match="must be larger"):
        validate_aggregation("1D", "60")

    with pytest.raises(ValueError, match="must be larger"):
        validate_aggregation("1W", "1D")


def __test_validate_aggregation_same_timeframe_fails__():
    """Same source and target must raise ValueError."""
    with pytest.raises(ValueError, match="must be larger"):
        validate_aggregation("60", "60")


def __test_validate_aggregation_not_divisible_fails__():
    """Non-divisible timeframes must raise ValueError."""
    with pytest.raises(ValueError, match="evenly divisible"):
        validate_aggregation("7", "60")  # 60 is not divisible by 7


def __test_validate_aggregation_calendar_targets__():
    """Weeks and months accept sources whose bars each lie inside one period."""
    validate_aggregation("1D", "1M")
    validate_aggregation("60", "1M")
    validate_aggregation("240", "1M")
    validate_aggregation("30S", "1M")
    validate_aggregation("1M", "3M")
    validate_aggregation("3M", "12M")
    validate_aggregation("1W", "2W")
    validate_aggregation("60", "1W")

    for source, target in (("1W", "1M"),  # a week straddles two months
                           ("2D", "1M"),  # so does a two-day bar
                           ("2M", "3M"),  # 2M bars do not tile 3M periods
                           ("7", "1W"),  # a 7-minute bar crosses midnight
                           ("2D", "2W")):
        with pytest.raises(ValueError, match="do not fit evenly"):
            validate_aggregation(source, target)


# --- Merge tests ---

def __test_merge_candles__():
    """Merge follows OHLCV rules: O=first, H=max, L=min, C=last, V=sum."""
    candles = [
        OHLCV(timestamp=60_000, open=10.0, high=15.0, low=8.0, close=12.0, volume=100.0),
        OHLCV(timestamp=120_000, open=12.0, high=18.0, low=11.0, close=14.0, volume=200.0),
        OHLCV(timestamp=180_000, open=14.0, high=16.0, low=9.0, close=13.0, volume=150.0),
    ]
    merged = _merge_candles(candles, bar_time=60_000)

    assert merged.timestamp == 60_000
    assert merged.open == 10.0
    assert merged.high == 18.0
    assert merged.low == 8.0
    assert merged.close == 13.0
    assert merged.volume == 450.0


def __test_merge_candles_volume_is_decimal_exact_sum__():
    """The merged volume is the exact decimal sum, not a binary float sum."""
    # BINANCE:BTCUSDT 30m volumes of one 150m period: a float sum gives 1495.4605800000002
    volumes = [271.27422, 484.71588, 387.2325, 199.30206, 152.93592]
    candles = [
        OHLCV(timestamp=i * 60_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=v)
        for i, v in enumerate(volumes)
    ]
    merged = _merge_candles(candles, bar_time=0)

    assert merged.volume == 1495.46058


# --- End-to-end aggregation tests ---

def __test_aggregate_5min_to_15min__(tmp_path):
    """5-minute candles aggregated to 15-minute should group by 3."""
    source = tmp_path / "test_5.ohlcv"
    target = tmp_path / "test_15.ohlcv"

    # Create 6 candles of 5-minute data (= 30 minutes = two 15-min bars)
    # Timestamps aligned to 5-min boundaries
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (Monday)
    candles = []
    for i in range(6):
        ts = base_ts + i * 300_000  # 5 min = 300 000 ms
        candles.append(OHLCV(
            timestamp=ts,
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=1000.0 + i * 10,
        ))

    _create_ohlcv_file(source, candles, "5")
    src_count, tgt_count = aggregate_ohlcv(source, target, "15")

    assert src_count == 6
    assert tgt_count == 2

    result = _read_all(target)
    assert len(result) == 2

    # First 15-min bar (candles 0,1,2) — use approx for float32 storage
    assert result[0].open == pytest.approx(100.0, rel=1e-5)
    assert result[0].high == pytest.approx(107.0, rel=1e-5)
    assert result[0].low == pytest.approx(95.0, rel=1e-5)
    assert result[0].close == pytest.approx(104.0, rel=1e-5)
    assert result[0].volume == pytest.approx(3030.0, rel=1e-5)

    # Second 15-min bar (candles 3,4,5)
    assert result[1].open == pytest.approx(103.0, rel=1e-5)
    assert result[1].high == pytest.approx(110.0, rel=1e-5)
    assert result[1].low == pytest.approx(98.0, rel=1e-5)
    assert result[1].close == pytest.approx(107.0, rel=1e-5)
    assert result[1].volume == pytest.approx(3120.0, rel=1e-5)


def __test_aggregate_daily_to_weekly__(tmp_path):
    """Daily candles should aggregate into weekly bars starting on Monday."""
    source = tmp_path / "test_1D.ohlcv"
    target = tmp_path / "test_1W.ohlcv"

    # 2024-01-01 is Monday — create 10 days (= 1 full week + 3 days)
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC (Monday)
    day = 86_400_000
    candles = []
    for i in range(10):
        candles.append(OHLCV(
            timestamp=base_ts + i * day,
            open=1.1000 + i * 0.001,
            high=1.1050 + i * 0.001,
            low=1.0950 + i * 0.001,
            close=1.1020 + i * 0.001,
            volume=50000.0 + i * 100,
        ))

    _create_ohlcv_file(source, candles, "1D")
    src_count, tgt_count = aggregate_ohlcv(source, target, "1W")

    assert src_count == 10
    assert tgt_count == 2  # 7 days + 3 days

    result = _read_all(target)
    assert len(result) == 2

    # float32 storage loses precision — use approx for comparisons
    # First weekly bar should have 7 candles
    assert result[0].open == pytest.approx(candles[0].open, rel=1e-5)
    assert result[0].close == pytest.approx(candles[6].close, rel=1e-5)
    assert result[0].volume == pytest.approx(sum(c.volume for c in candles[:7]), rel=1e-5)

    # Second weekly bar should have 3 candles
    assert result[1].open == pytest.approx(candles[7].open, rel=1e-5)
    assert result[1].close == pytest.approx(candles[9].close, rel=1e-5)
    assert result[1].volume == pytest.approx(sum(c.volume for c in candles[7:]), rel=1e-5)


def __test_aggregate_daily_to_monthly__(tmp_path):
    """Daily candles group into calendar months of their own length."""
    source = tmp_path / "test_1D.ohlcv"
    target = tmp_path / "test_1M.ohlcv"

    # 2024-01-01 .. 2024-03-31: 31 + 29 (leap February) + 31 days
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
    day = 86_400_000
    candles = [OHLCV(timestamp=base_ts + i * day, open=100.0 + i, high=101.0 + i,
                     low=99.0 + i, close=100.5 + i, volume=10.0)
               for i in range(91)]

    _create_ohlcv_file(source, candles, "1D")
    src_count, tgt_count = aggregate_ohlcv(source, target, "1M", source_tf="1D")

    assert src_count == 91
    assert tgt_count == 3

    result = _read_all(target)
    month_starts = [base_ts, base_ts + 31 * day, base_ts + 60 * day]
    month_lengths = [31, 29, 31]
    first = 0
    for bar, start, length in zip(result, month_starts, month_lengths):
        window = candles[first:first + length]
        assert bar.timestamp == start
        assert bar.open == pytest.approx(window[0].open, rel=1e-5)
        assert bar.close == pytest.approx(window[-1].close, rel=1e-5)
        assert bar.high == pytest.approx(max(c.high for c in window), rel=1e-5)
        assert bar.low == pytest.approx(min(c.low for c in window), rel=1e-5)
        assert bar.volume == pytest.approx(10.0 * length, rel=1e-5)
        first += length


def __test_aggregate_hourly_to_monthly__(tmp_path):
    """Intraday candles split at the month boundary, not at a fixed month length."""
    source = tmp_path / "test_60.ohlcv"
    target = tmp_path / "test_1M.ohlcv"

    # 2024-02-28 00:00 .. 2024-03-02 23:00 UTC: two leap-February days, two March days
    base_ts = 1_709_078_400_000  # 2024-02-28 00:00:00 UTC
    hour = 3_600_000
    candles = [OHLCV(timestamp=base_ts + i * hour, open=1.0, high=1.0, low=1.0,
                     close=1.0, volume=1.0)
               for i in range(96)]

    _create_ohlcv_file(source, candles, "60")
    src_count, tgt_count = aggregate_ohlcv(source, target, "1M", source_tf="60")

    assert src_count == 96
    assert tgt_count == 2

    result = _read_all(target)
    assert result[0].timestamp == 1_706_745_600_000  # 2024-02-01 00:00:00 UTC
    assert result[0].volume == pytest.approx(48.0)
    assert result[1].timestamp == 1_709_251_200_000  # 2024-03-01 00:00:00 UTC
    assert result[1].volume == pytest.approx(48.0)


def __test_aggregate_preserves_high_low__(tmp_path):
    """High/Low must be the true extremes across all source candles."""
    source = tmp_path / "test_src.ohlcv"
    target = tmp_path / "test_tgt.ohlcv"

    # Two target bars, so both the merge and the bar boundary are exercised
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
    step = 300_000  # 5 min
    candles = [
        # First 15-min bar: extreme high in candle 1, extreme low in candle 2
        OHLCV(timestamp=base_ts,            open=100.0, high=110.0, low=90.0, close=105.0, volume=100.0),
        OHLCV(timestamp=base_ts + step,     open=105.0, high=120.0, low=95.0, close=115.0, volume=200.0),
        OHLCV(timestamp=base_ts + 2 * step, open=115.0, high=116.0, low=85.0, close=100.0, volume=150.0),
        # Second 15-min bar
        OHLCV(timestamp=base_ts + 3 * step, open=100.0, high=105.0, low=92.0, close=98.0, volume=80.0),
        OHLCV(timestamp=base_ts + 4 * step, open=98.0, high=130.0, low=88.0, close=125.0, volume=300.0),
        OHLCV(timestamp=base_ts + 5 * step, open=125.0, high=126.0, low=91.0, close=95.0, volume=120.0),
    ]

    _create_ohlcv_file(source, candles, "5")
    aggregate_ohlcv(source, target, "15")

    result = _read_all(target)
    assert len(result) == 2

    # First bar: high from candle 1 (120), low from candle 2 (85)
    assert result[0].high == pytest.approx(120.0, rel=1e-5)
    assert result[0].low == pytest.approx(85.0, rel=1e-5)

    # Second bar: high from candle 4 (130), low from candle 5 (88)
    assert result[1].high == pytest.approx(130.0, rel=1e-5)
    assert result[1].low == pytest.approx(88.0, rel=1e-5)


def __test_aggregate_session_anchored__():
    """Off-grid session (09:30) anchors 60m target bars to the session open."""
    from datetime import datetime, time
    from zoneinfo import ZoneInfo
    from pynecore.core.syminfo import SymInfoSession

    ny = ZoneInfo("America/New_York")

    def ts(h: int, mi: int) -> int:
        return int(datetime(2024, 1, 17, h, mi, tzinfo=ny).timestamp()) * 1000

    # 30m source candles across a 09:30-opening session.
    candles = [
        OHLCV(timestamp=ts(9, 30),  open=10.0, high=11.0, low=9.0,  close=10.5, volume=1.0),
        OHLCV(timestamp=ts(10, 0),  open=10.5, high=12.0, low=10.0, close=11.5, volume=2.0),
        OHLCV(timestamp=ts(10, 30), open=11.5, high=13.0, low=11.0, close=12.0, volume=3.0),
        OHLCV(timestamp=ts(11, 0),  open=12.0, high=12.5, low=11.5, close=12.2, volume=4.0),
    ]
    session_starts = [SymInfoSession(day=d, time=time(9, 30, 0)) for d in range(5)]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "src.ohlcv"
        target = Path(tmp) / "tgt.ohlcv"
        _create_ohlcv_file(source, candles, "30")
        aggregate_ohlcv(source, target, "60", tz=ny, session_starts=session_starts)
        result = _read_all(target)

    # Two 60m bars anchored to the session: 09:30 (09:30+10:00) and 10:30 (10:30+11:00).
    assert [c.timestamp for c in result] == [ts(9, 30), ts(10, 30)]
    assert result[0].volume == pytest.approx(3.0)   # 1 + 2
    assert result[1].volume == pytest.approx(7.0)   # 3 + 4


def __test_aggregate_observed_holiday_grouping__():
    """'Observed' symbols group by trading days present — holidays consume no slot."""
    from datetime import datetime, time, timedelta, date
    from zoneinfo import ZoneInfo
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    ny = ZoneInfo("America/New_York")
    t17 = time(17, 0)
    hours = [SymInfoInterval(day=d, start=t17, end=t17) for d in (6, 0, 1, 2, 3)]
    starts = [SymInfoSession(day=d, time=t17) for d in (6, 0, 1, 2, 3)]

    def open_ts(trading_day: date) -> int:
        prev = trading_day - timedelta(days=1)
        return int(datetime(prev.year, prev.month, prev.day, 17, tzinfo=ny).timestamp()) * 1000

    # Two weeks of daily bars; Mon Jan 13 2025 is a holiday (no bar)
    days = [date(2025, 1, d) for d in (6, 7, 8, 9, 10, 14, 15, 16, 17)]
    candles = [
        OHLCV(timestamp=open_ts(d), open=10.0, high=11.0, low=9.0, close=10.5,
              volume=float(i + 1))
        for i, d in enumerate(days)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "src.ohlcv"
        target = Path(tmp) / "tgt.ohlcv"
        _create_ohlcv_file(source, candles, "1D")
        aggregate_ohlcv(source, target, "2D", tz=ny, session_starts=starts,
                        opening_hours=hours, sym_type='futures', source_tf='1D')
        result = _read_all(target)

    # Jan 6 is weekday ordinal 3 (odd), so it pairs with the dataless Jan 5
    # slot; afterwards the observed days pair up — the Jan 13 holiday consumes
    # no slot, so Jan 14+15 and Jan 16+17 group together (TradingView grid)
    expected_groups = [
        [date(2025, 1, 6)],
        [date(2025, 1, 7), date(2025, 1, 8)],
        [date(2025, 1, 9), date(2025, 1, 10)],
        [date(2025, 1, 14), date(2025, 1, 15)],
        [date(2025, 1, 16), date(2025, 1, 17)],
    ]
    assert [c.timestamp for c in result] == [open_ts(g[0]) for g in expected_groups]
    expected_vols = [sum(float(days.index(d) + 1) for d in g) for g in expected_groups]
    assert [c.volume for c in result] == pytest.approx(expected_vols)


def __test_aggregate_observed_multi_day_source__(tmp_path):
    """An observed 2D source chained to 4D groups exactly like 1D aggregated to 4D."""
    from datetime import datetime, time, timedelta, date
    from zoneinfo import ZoneInfo
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    ny = ZoneInfo("America/New_York")
    t17 = time(17, 0)
    hours = [SymInfoInterval(day=d, start=t17, end=t17) for d in (6, 0, 1, 2, 3)]
    starts = [SymInfoSession(day=d, time=t17) for d in (6, 0, 1, 2, 3)]

    def open_ts(trading_day: date) -> int:
        prev = trading_day - timedelta(days=1)
        return int(datetime(prev.year, prev.month, prev.day, 17, tzinfo=ny).timestamp()) * 1000

    # January 2025 weekdays from Jan 6, without the Jan 13 and Jan 20 holidays
    days = [date(2025, 1, d) for d in range(6, 32)
            if date(2025, 1, d).weekday() < 5 and d not in (13, 20)]
    candles = [OHLCV(timestamp=open_ts(d), open=10.0, high=11.0, low=9.0, close=10.5,
                     volume=float(i + 1))
               for i, d in enumerate(days)]

    daily = tmp_path / "src_1D.ohlcv"
    _create_ohlcv_file(daily, candles, "1D")
    kwargs = dict(tz=ny, session_starts=starts, opening_hours=hours, sym_type='futures')

    aggregate_ohlcv(daily, tmp_path / "direct_4D.ohlcv", "4D", source_tf='1D', **kwargs)
    aggregate_ohlcv(daily, tmp_path / "mid_2D.ohlcv", "2D", source_tf='1D', **kwargs)
    aggregate_ohlcv(tmp_path / "mid_2D.ohlcv", tmp_path / "chained_4D.ohlcv", "4D",
                    source_tf='2D', **kwargs)

    direct = _read_all(tmp_path / "direct_4D.ohlcv")
    chained = _read_all(tmp_path / "chained_4D.ohlcv")
    assert len(direct) > 3
    assert [c.timestamp for c in chained] == [c.timestamp for c in direct]
    assert [c.volume for c in chained] == pytest.approx([c.volume for c in direct])


def __test_aggregate_monthly_resolves_intraday_bar_by_last_instant__(tmp_path):
    """A source bar holding the session open starts the new month; a session end cuts it."""
    from datetime import datetime, time
    from zoneinfo import ZoneInfo
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    ny = ZoneInfo("America/New_York")
    starts = [SymInfoSession(day=d, time=time(17, 5)) for d in (6, 0, 1, 2, 3)]
    hours = [SymInfoInterval(day=d, start=time(17, 5), end=time(17, 0))
             for d in (6, 0, 1, 2, 3)]

    def ts(day: int, hour: int) -> int:
        return int(datetime(2024, 1, day, hour, tzinfo=ny).timestamp()) * 1000

    # A 4-hour UTC-clock grid around the January 31 17:05 open of trading day Feb 1:
    # the 15:00 bar runs to 19:00 and holds that open
    candles = [OHLCV(timestamp=ts(31, h), open=1.0, high=1.0, low=1.0, close=1.0,
                     volume=1.0)
               for h in (7, 11, 15, 19, 23)]
    source = tmp_path / "src_240.ohlcv"
    _create_ohlcv_file(source, candles, "240")

    # Without opening hours the bar ends at its nominal close: it belongs to February
    aggregate_ohlcv(source, tmp_path / "nominal.ohlcv", "1M", tz=ny, session_starts=starts,
                    sym_type='forex', source_tf='240')
    nominal = _read_all(tmp_path / "nominal.ohlcv")
    assert [c.volume for c in nominal] == pytest.approx([2.0, 3.0])
    assert nominal[1].timestamp == int(datetime(2024, 1, 31, 17, 5, tzinfo=ny).timestamp()) * 1000

    # The session ending at 17:00 closes the bar before the open: it stays in January
    aggregate_ohlcv(source, tmp_path / "cut.ohlcv", "1M", tz=ny, session_starts=starts,
                    opening_hours=hours, sym_type='forex', source_tf='240')
    cut = _read_all(tmp_path / "cut.ohlcv")
    assert [c.volume for c in cut] == pytest.approx([3.0, 2.0])


def __test_aggregate_observed_without_sessions_warns__(tmp_path, caplog):
    """An observed multi-period grid without session metadata says it groups by date."""
    day_ms = 86_400_000
    candles = [
        OHLCV(timestamp=1_736_121_600_000 + i * day_ms, open=10.0, high=11.0, low=9.0,
              close=10.5, volume=1.0)
        for i in range(4)
    ]
    source = tmp_path / "src.ohlcv"
    _create_ohlcv_file(source, candles, "1D")

    with caplog.at_level("WARNING", logger="pynecore.core.aggregator"):
        aggregate_ohlcv(source, tmp_path / "tgt.ohlcv", "2D", sym_type='futures',
                        source_tf='1D')
    assert "grouped by calendar date" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING", logger="pynecore.core.aggregator"):
        aggregate_ohlcv(source, tmp_path / "tgt.ohlcv", "2D", sym_type='crypto',
                        source_tf='1D')
    assert "grouped by calendar date" not in caplog.text


@pytest.mark.skipif(not hasattr(time, 'tzset'), reason="time.tzset() is not available on Windows")
def __test_aggregate_default_tz_is_utc_not_host__(tmp_path, monkeypatch):
    """Without an explicit tz the day boundaries are UTC, whatever the host zone is."""
    hour_ms = 3_600_000
    # 2025-01-06 00:00 UTC onwards, 48 hourly bars
    candles = [
        OHLCV(timestamp=1_736_121_600_000 + i * hour_ms, open=10.0, high=11.0, low=9.0,
              close=10.5, volume=1.0)
        for i in range(48)
    ]
    source = tmp_path / "src.ohlcv"
    _create_ohlcv_file(source, candles, "60")

    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        aggregate_ohlcv(source, tmp_path / "tgt.ohlcv", "1D", source_tf='60')
        result = _read_all(tmp_path / "tgt.ohlcv")
    finally:
        monkeypatch.undo()
        time.tzset()

    assert [c.timestamp for c in result] == [1_736_121_600_000, 1_736_208_000_000]
    assert [c.volume for c in result] == pytest.approx([24.0, 24.0])


def __test_aggregate_rejects_source_as_target__(tmp_path):
    """Aggregating onto the source file (directly or via a hard link) fails and keeps it."""
    candles = [
        OHLCV(timestamp=1_736_121_600_000 + i * 3_600_000, open=10.0, high=11.0, low=9.0,
              close=10.5, volume=1.0)
        for i in range(48)
    ]
    source = tmp_path / "src.ohlcv"
    _create_ohlcv_file(source, candles, "60")
    before = source.read_bytes()
    alias = tmp_path / "alias.ohlcv"
    alias.hardlink_to(source)

    for target in (source, alias):
        with pytest.raises(ValueError, match="is the source file"):
            aggregate_ohlcv(source, target, "1D", source_tf='60')
    assert source.read_bytes() == before


# --- Single-record / empty-source edge cases (issue #70) ---

def __test_aggregate_single_record_floors_to_period__(tmp_path):
    """A one-record source aggregates to ONE bar floored onto the period grid.

    That lone bar IS a whole target period and must be emitted with a
    period-boundary timestamp, not the raw sub-bar instant. Without this, a finer
    one-period ``--security`` feed would expose a daily bar stamped mid-day,
    shifting single-period D/W/M confirmation and ``request.security(.., time)``
    by up to a full period (issue #70 edge case).
    """
    from datetime import timezone

    source = tmp_path / "one.ohlcv"
    target = tmp_path / "one_1D.ohlcv"

    # A single 15-minute bar at 2024-01-02 14:30 UTC (mid-day, NOT civil midnight)
    raw_ts = 1_704_205_800_000  # 2024-01-02 14:30:00 UTC
    _create_ohlcv_file(source, [
        OHLCV(timestamp=raw_ts, open=100.0, high=110.0, low=90.0, close=105.0, volume=7.0),
    ], "15")

    # The lone bar floors via the SAME ``get_bar_time`` call as the multi-bar loop,
    # so it is consistent with a normal feed's stamping; pin UTC for a clean
    # civil-midnight expectation.
    src_count, tgt_count = aggregate_ohlcv(source, target, "1D", tz=timezone.utc,
                                           source_tf="15")
    assert src_count == 1
    assert tgt_count == 1

    # Output: exactly one bar, stamped at the civil daily open (00:00 UTC), with
    # the lone bar's OHLCV preserved.
    with OHLCVReader(target) as reader:
        assert reader.size == 1
        bar = reader.read(0)
    assert bar.timestamp == 1_704_153_600_000  # 2024-01-02 00:00:00 UTC
    assert bar.open == pytest.approx(100.0, rel=1e-5)
    assert bar.high == pytest.approx(110.0, rel=1e-5)
    assert bar.low == pytest.approx(90.0, rel=1e-5)
    assert bar.close == pytest.approx(105.0, rel=1e-5)
    assert bar.volume == pytest.approx(7.0, rel=1e-5)


def __test_aggregate_empty_source_stays_empty__(tmp_path):
    """A zero-record source aggregates to zero bars — the single-record fast path
    must NOT fire for an empty file."""
    source = tmp_path / "empty.ohlcv"
    target = tmp_path / "empty_1D.ohlcv"
    _create_ohlcv_file(source, [], "15")

    src_count, tgt_count = aggregate_ohlcv(source, target, "1D", source_tf="15")
    assert src_count == 0
    assert tgt_count == 0
