"""
Reopening an OHLCV file for append must use the real last timestamp it holds.

A calendar-monthly feed sits on no fixed grid: consecutive bars are 28 to 31 days
apart, so no single interval describes the series. A BINANCE:BTCUSDT monthly file
once reported "Got 1782864000 after 9906364800" on a perfectly ordered append — a
year-2283 timestamp that no record in the file held, extrapolated from a 31-day
first interval over 3138 records.

The v2 header states ``first_timestamp`` and ``last_timestamp`` outright, so the
append guard compares against the file's own extent, never against an arithmetic
projection, and irregular spacing is not the writer's business. The declared
period stays purely nominal metadata.
"""
from datetime import datetime, UTC
from pathlib import Path

import pytest

from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.ohlcv import OHLCVReader
from pynecore.types.ohlcv import OHLCV


def _month_starts(count: int) -> list[int]:
    """UTC midnight of the first day of ``count`` consecutive months, in ms."""
    out, dt = [], datetime(2024, 1, 1, tzinfo=UTC)
    for _ in range(count):
        out.append(int(dt.timestamp()) * 1000)
        dt = dt.replace(year=dt.year + dt.month // 12, month=dt.month % 12 + 1)
    return out


def _write_months(path: Path, stamps: list[int]) -> None:
    with OHLCVWriter(path, "1M", truncate=True) as writer:
        for ts in stamps:
            writer.write(OHLCV(ts, 1.0, 2.0, 0.5, 1.5, 100.0))


def __test_append_to_monthly_file_is_accepted__(tmp_path: Path, log):
    """The month after the file's last month appends instead of being rejected"""
    stamps = _month_starts(14)
    path = tmp_path / "monthly.ohlcv"
    _write_months(path, stamps[:-1])

    with OHLCVWriter(path, "1M") as writer:
        writer.write(OHLCV(stamps[-1], 1.0, 2.0, 0.5, 1.5, 100.0))

    with OHLCVReader(str(path)) as reader:
        assert reader.end_timestamp == stamps[-1]


def __test_reopened_writer_sees_the_real_last_bar__(tmp_path: Path, log):
    """The append guard compares against a timestamp the file actually holds"""
    stamps = _month_starts(14)
    path = tmp_path / "monthly_probe.ohlcv"
    _write_months(path, stamps[:-1])

    with OHLCVReader(str(path)) as reader:
        on_disk = reader.end_timestamp

    writer = OHLCVWriter(path, "1M")
    writer.open()
    try:
        # noinspection PyProtectedMember
        assert writer._last_timestamp == on_disk
    finally:
        writer.close()


def __test_out_of_order_append_still_rejected__(tmp_path: Path, log):
    """The guard keeps its job: a genuinely older bar is refused"""
    stamps = _month_starts(4)
    path = tmp_path / "monthly_reject.ohlcv"
    _write_months(path, stamps)

    with OHLCVReader(str(path)) as reader:
        first_on_disk = reader.start_timestamp

    with pytest.raises(ValueError, match="strictly increasing"):
        with OHLCVWriter(path, "1M") as writer:
            writer.write(OHLCV(first_on_disk, 1.0, 2.0, 0.5, 1.5, 100.0))


def __test_calendar_month_spacing_is_not_declared_dense__(tmp_path: Path, log):
    """Irregular calendar spacing leaves the file marked sparse, not dense"""
    stamps = _month_starts(6)
    path = tmp_path / "monthly_dense.ohlcv"
    _write_months(path, stamps)

    with OHLCVReader(str(path)) as reader:
        assert reader.period == "1M"
        assert reader.dense is False
