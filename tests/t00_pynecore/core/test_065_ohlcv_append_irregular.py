"""
Reopening an OHLCV file for append must read its real last timestamp.

``OHLCVWriter.open`` used to derive it arithmetically, as
``first + interval * (size - 1)`` with ``interval`` measured between the first
two records. That only holds for a file whose records sit on an exact grid.
A monthly BINANCE:BTCUSDT feed reported "Got 1782864000 after 9906364800" on a
perfectly ordered append -- a timestamp in the year 2283 that no record in the
file held, extrapolated from a 31-day first interval over 3138 records.

(The fixed-grid model itself is a separate, open limitation: gap filling and
``seek_to_timestamp`` both index by a single interval, which cannot express
28-to-31-day calendar months. These tests cover the append guard only.)
"""
from datetime import datetime, UTC
from pathlib import Path

import pytest

from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
from pynecore.types.ohlcv import OHLCV


def _month_starts(count: int) -> list[int]:
    """UTC midnight of the first day of ``count`` consecutive months."""
    out, dt = [], datetime(2024, 1, 1, tzinfo=UTC)
    for _ in range(count):
        out.append(int(dt.timestamp()))
        dt = dt.replace(year=dt.year + dt.month // 12, month=dt.month % 12 + 1)
    return out


def _write_months(path: Path, stamps: list[int]) -> None:
    with OHLCVWriter(path, truncate=True) as writer:
        for ts in stamps:
            writer.write(OHLCV(ts, 1.0, 2.0, 0.5, 1.5, 100.0))


def __test_append_to_monthly_file_is_accepted__(tmp_path: Path, log):
    """The month after the file's last month appends instead of being rejected"""
    stamps = _month_starts(14)
    path = tmp_path / "monthly.ohlcv"
    _write_months(path, stamps[:-1])

    with OHLCVWriter(path) as writer:
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

    writer = OHLCVWriter(path)
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

    with pytest.raises(ValueError, match="chronological order"):
        with OHLCVWriter(path) as writer:
            writer.write(OHLCV(first_on_disk, 1.0, 2.0, 0.5, 1.5, 100.0))
