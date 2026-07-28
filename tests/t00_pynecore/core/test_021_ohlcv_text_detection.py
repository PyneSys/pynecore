"""
Binary OHLCV must not be mistaken for a text file.

``OHLCVReader.open`` refuses a CSV saved under an ``.ohlcv`` name. The check used
to read 32 bytes and reject anything that decoded as ASCII — but a perfectly
valid record decodes as ASCII too whenever every one of its bytes falls below
0x80, which is ordinary for a small price with no volume. A BIST:PGSUS hourly
export hit exactly that and became unreadable. A record file is always a whole
number of ``RECORD_SIZE`` records, so the length is what settles it.
"""
import struct
from pathlib import Path

import pytest

from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter, RECORD_SIZE, STRUCT_FORMAT
from pynecore.types.ohlcv import OHLCV


def __test_all_ascii_records_are_readable__(tmp_path: Path, log):
    """A record whose every byte is below 0x80 is still binary, not text"""
    # Timestamp and prices chosen so the packed record holds no byte >= 0x80 —
    # the same shape as the export that triggered this (0x67076D60 packs to
    # b'`m\x07g', and zero prices pack to zero bytes).
    ts = 0x67076D60  # 2024-10-09 18:45:52 UTC
    record = struct.pack(STRUCT_FORMAT, ts, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert max(record) < 0x80, "fixture must be all-ASCII to exercise the check"

    path = tmp_path / "ascii_records.ohlcv"
    path.write_bytes(record * 3)

    with OHLCVReader(str(path)) as reader:
        assert reader.size == 3
        assert reader.start_timestamp == ts


def __test_csv_under_ohlcv_name_still_rejected__(tmp_path: Path, log):
    """A CSV saved as .ohlcv is still caught, with the convert-from hint"""
    path = tmp_path / "actually_a_csv.ohlcv"
    path.write_text("time,open,high,low,close,volume\n"
                    "2025-01-01T00:00:00Z,1,2,0.5,1.5,100\n", encoding="utf-8")
    assert path.stat().st_size % RECORD_SIZE != 0

    with pytest.raises(ValueError, match="Text file detected"):
        with OHLCVReader(str(path)):
            pass


def __test_written_file_round_trips__(tmp_path: Path, log):
    """The writer's own output always reads back — the guard never fires on it"""
    path = tmp_path / "written.ohlcv"
    with OHLCVWriter(path, truncate=True) as writer:
        for i in range(4):
            writer.write(OHLCV(1735689600 + i * 3600, 0.0, 0.0, 0.0, 0.0, 0.0))

    with OHLCVReader(str(path)) as reader:
        assert reader.size == 4
