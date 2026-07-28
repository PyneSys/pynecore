"""
Legacy v1 OHLCV files must not be mistaken for text files.

A v1 file carries no magic bytes, so ``OHLCVReader.open`` falls back to the legacy
reader for anything that is not v2, and only that reader runs the text probe. The
probe used to read 32 bytes and reject whatever decoded as ASCII — but a perfectly
valid v1 record decodes as ASCII too whenever every one of its bytes falls below
0x80, which is ordinary for a small price with no volume. A BIST:PGSUS hourly
export hit exactly that and became unreadable. A v1 file is always a whole number
of ``RECORD_SIZE`` records, so the length is what settles it.

The v1 writer is gone, so the fixtures below hand-pack v1 records. That is why this
module imports the v1 format constants from ``pynecore.core.ohlcv_legacy`` — a
licence reserved for the v1 backward-compatibility tests, never for production,
plugin or example code.
"""
import struct
from pathlib import Path

import pytest

from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter
from pynecore.core.ohlcv_legacy import RECORD_SIZE, STRUCT_FORMAT
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
        assert reader.start_timestamp == ts * 1000


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
    """A freshly written v2 file is magic-dispatched, so the probe never sees it"""
    path = tmp_path / "written.ohlcv"
    with OHLCVWriter(path, "60", truncate=True) as writer:
        for i in range(4):
            writer.write(OHLCV(1_735_689_600_000 + i * 3_600_000, 0.0, 0.0, 0.0, 0.0, 0.0))

    with OHLCVReader(str(path)) as reader:
        assert reader.size == 4
