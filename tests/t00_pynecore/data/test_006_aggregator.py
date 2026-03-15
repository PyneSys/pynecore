"""
@pyne
"""
import pytest
import tempfile
from pathlib import Path

from pynecore.core.aggregator import validate_aggregation, aggregate_ohlcv, _merge_candles
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader
from pynecore.types.ohlcv import OHLCV


def main():
    pass


def _create_ohlcv_file(path: Path, candles: list[OHLCV]) -> None:
    """Helper to write candles into an .ohlcv file."""
    with OHLCVWriter(path, truncate=True) as writer:
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


# --- Merge tests ---

def __test_merge_candles__():
    """Merge follows OHLCV rules: O=first, H=max, L=min, C=last, V=sum."""
    candles = [
        OHLCV(timestamp=1000, open=10.0, high=15.0, low=8.0, close=12.0, volume=100.0),
        OHLCV(timestamp=1060, open=12.0, high=18.0, low=11.0, close=14.0, volume=200.0),
        OHLCV(timestamp=1120, open=14.0, high=16.0, low=9.0, close=13.0, volume=150.0),
    ]
    merged = _merge_candles(candles, bar_time=1000)

    assert merged.timestamp == 1000
    assert merged.open == 10.0
    assert merged.high == 18.0
    assert merged.low == 8.0
    assert merged.close == 13.0
    assert merged.volume == 450.0


# --- End-to-end aggregation tests ---

def __test_aggregate_5min_to_15min__(tmp_path):
    """5-minute candles aggregated to 15-minute should group by 3."""
    source = tmp_path / "test_5.ohlcv"
    target = tmp_path / "test_15.ohlcv"

    # Create 6 candles of 5-minute data (= 30 minutes = two 15-min bars)
    # Timestamps aligned to 5-min boundaries
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC (Monday)
    candles = []
    for i in range(6):
        ts = base_ts + i * 300  # 5 min = 300 sec
        candles.append(OHLCV(
            timestamp=ts,
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=1000.0 + i * 10,
        ))

    _create_ohlcv_file(source, candles)
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
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC (Monday)
    day = 86400
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

    _create_ohlcv_file(source, candles)
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


def __test_aggregate_preserves_high_low__(tmp_path):
    """High/Low must be the true extremes across all source candles."""
    source = tmp_path / "test_src.ohlcv"
    target = tmp_path / "test_tgt.ohlcv"

    # Need at least 2 target bars (OHLCVReader requires 2+ candles for interval)
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    candles = [
        # First 15-min bar: extreme high in candle 1, extreme low in candle 2
        OHLCV(timestamp=base_ts,       open=100.0, high=110.0, low=90.0,  close=105.0, volume=100.0),
        OHLCV(timestamp=base_ts + 300,  open=105.0, high=120.0, low=95.0,  close=115.0, volume=200.0),
        OHLCV(timestamp=base_ts + 600,  open=115.0, high=112.0, low=85.0,  close=100.0, volume=150.0),
        # Second 15-min bar
        OHLCV(timestamp=base_ts + 900,  open=100.0, high=105.0, low=92.0,  close=98.0, volume=80.0),
        OHLCV(timestamp=base_ts + 1200, open=98.0,  high=130.0, low=88.0,  close=125.0, volume=300.0),
        OHLCV(timestamp=base_ts + 1500, open=125.0, high=126.0, low=91.0,  close=95.0, volume=120.0),
    ]

    _create_ohlcv_file(source, candles)
    aggregate_ohlcv(source, target, "15")

    result = _read_all(target)
    assert len(result) == 2

    # First bar: high from candle 1 (120), low from candle 2 (85)
    assert result[0].high == pytest.approx(120.0, rel=1e-5)
    assert result[0].low == pytest.approx(85.0, rel=1e-5)

    # Second bar: high from candle 4 (130), low from candle 5 (88)
    assert result[1].high == pytest.approx(130.0, rel=1e-5)
    assert result[1].low == pytest.approx(88.0, rel=1e-5)
