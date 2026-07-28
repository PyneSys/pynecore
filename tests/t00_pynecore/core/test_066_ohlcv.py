"""OHLCV v2 binary storage tests."""

import json
import math
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

import pytest

import pynecore.core.ohlcv as ohlcv
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter, record_count
from pynecore.types.ohlcv import OHLCV

_MAGIC = b"\x89PYN\r\n\x1a\n"
_HEADER = struct.Struct("<8sHHIIHHQqqIB3x8x")
_DESCRIPTOR = struct.Struct("<BBBxH18s")
_HEADER_SIZE = 64 + 6 * 24
_DTYPE_SIZE = {2: 8, 5: 4, 6: 8}


def _write_candles(
    path: Path,
    candles: list[OHLCV],
    period: str = "1",
    *,
    mintick: float | None = None,
) -> None:
    with OHLCVWriter(path, period, mintick=mintick, truncate=True) as writer:
        for candle in candles:
            writer.write(candle)


def _header(path: Path) -> tuple[bytes, int, int, int, int, int, int, int, int, int, int, int]:
    return _HEADER.unpack(path.read_bytes()[:64])


def _record_size(path: Path) -> int:
    return _header(path)[4]


def _assert_candle_close(actual: OHLCV, expected: OHLCV, mintick: float) -> None:
    assert actual.timestamp == expected.timestamp
    assert actual.open == expected.open
    assert actual.volume == expected.volume
    for actual_price, expected_price in zip(actual[2:5], expected[2:5], strict=True):
        assert abs(actual_price - expected_price) < mintick / 2


def _raw_descriptor(
    role: int, dtype: int, base: int, byte_offset: int, name: str
) -> tuple[int, int, int, int, str]:
    return role, dtype, base, byte_offset, name


def _write_raw_file(
    path: Path,
    columns: tuple[tuple[int, int, int, int, str], ...],
    records: list[bytes],
    *,
    interval_value: int = 1,
    interval_unit: int = 2,
    flags: int = 0,
) -> None:
    header_size = 64 + len(columns) * 24
    record_size = sum(_DTYPE_SIZE[column[1]] for column in columns)
    timestamp_offset = next(column[3] for column in columns if column[0] == 0)
    first_timestamp = (
        struct.unpack_from("<q", records[0], timestamp_offset)[0] if records else 0
    )
    last_timestamp = (
        struct.unpack_from("<q", records[-1], timestamp_offset)[0] if records else 0
    )
    header = _HEADER.pack(
        _MAGIC,
        2,
        0,
        header_size,
        record_size,
        len(columns),
        flags,
        len(records),
        first_timestamp,
        last_timestamp,
        interval_value,
        interval_unit,
    )
    descriptors = b"".join(
        _DESCRIPTOR.pack(role, dtype, base, offset, name.encode("ascii").ljust(18, b"\0"))
        for role, dtype, base, offset, name in columns
    )
    path.write_bytes(header + descriptors + b"".join(records))


def _mutate(path: Path, offset: int, replacement: bytes) -> None:
    data = bytearray(path.read_bytes())
    data[offset : offset + len(replacement)] = replacement
    path.write_bytes(data)


@pytest.mark.parametrize(
    ("name", "mintick", "candles"),
    [
        (
            "btc",
            0.01,
            [
                OHLCV(1_735_689_600_000, 100_000.0, 100_123.45, 99_876.54, 100_050.01, 12.5),
                OHLCV(1_735_689_660_000, 100_050.0, 100_200.02, 99_999.99, 100_111.11, 13.5),
            ],
        ),
        (
            "eurusd",
            0.00001,
            [
                OHLCV(1_735_689_600_000, 1.08, 1.08123, 1.07987, 1.08045, 1_000_000.0),
                OHLCV(1_735_689_660_000, 1.08045, 1.08234, 1.08001, 1.08111, 900_000.0),
            ],
        ),
    ],
)
def __test_default_profile_round_trip__(
    tmp_path: Path, name: str, mintick: float, candles: list[OHLCV]
):
    """The 36-byte default profile preserves realistic prices within half a tick."""
    path = tmp_path / f"{name}.ohlcv"
    _write_candles(path, candles)

    assert _record_size(path) == 36
    with OHLCVReader(path) as reader:
        assert reader.period == "1"
        assert reader.size == len(candles)
        assert reader.dense
        for actual, expected in zip(reader, candles, strict=True):
            _assert_candle_close(actual, expected, mintick)


def __test_byte_exact_default_layout__(tmp_path: Path):
    """Header, descriptors, and record bytes pin the little-endian packed wire layout."""
    path = tmp_path / "layout.ohlcv"
    candle = OHLCV(1_000, 100.0, 101.5, 98.0, 100.25, 10.0)
    _write_candles(path, [candle])
    raw = path.read_bytes()

    assert raw[:8] == b"\x89PYN\r\n\x1a\n"
    assert raw[8:64] == bytes.fromhex(
        "02000000d00000002400000006000000"
        "0100000000000000e803000000000000"
        "e8030000000000000100000002000000"
        "0000000000000000"
    )
    assert len(raw) == 208 + 36

    descriptors = [
        _DESCRIPTOR.unpack_from(raw, 64 + index * 24) for index in range(6)
    ]
    assert [(role, dtype, base, offset) for role, dtype, base, offset, _ in descriptors] == [
        (0, 2, 255, 0),
        (1, 6, 255, 8),
        (2, 5, 1, 16),
        (3, 5, 1, 20),
        (4, 5, 1, 24),
        (5, 6, 255, 28),
    ]
    assert [name.rstrip(b"\0") for *_, name in descriptors] == [
        b"timestamp",
        b"open",
        b"high",
        b"low",
        b"close",
        b"volume",
    ]

    record = raw[208:]
    assert record == bytes.fromhex(
        "e8030000000000000000000000005940"
        "0000c03f000000c00000803e0000000000002440"
    )
    assert struct.unpack("<qdfffd", record) == (1_000, 100.0, 1.5, -2.0, 0.25, 10.0)


@pytest.mark.parametrize(
    ("name", "period", "timestamps", "target_index"),
    [
        (
            "dst",
            "1D",
            [0, 86_400_000, 169_200_000, 255_600_000],
            2,
        ),
        (
            "partial_close",
            "30",
            [0, 1_800_000, 2_700_000, 86_400_000],
            2,
        ),
        (
            "calendar_month",
            "1M",
            [
                int(datetime(2024, month, 1, tzinfo=UTC).timestamp() * 1000)
                for month in range(1, 5)
            ],
            2,
        ),
        (
            "bist_half_slot",
            "60",
            [0, 3_600_000, 59_400_000, 63_000_000],
            2,
        ),
    ],
)
def __test_off_grid_addressing_uses_exact_timestamps__(
    tmp_path: Path,
    name: str,
    period: str,
    timestamps: list[int],
    target_index: int,
):
    """Bisect finds real bars across DST, partial, monthly, and half-slot gaps."""
    path = tmp_path / f"{name}.ohlcv"
    candles = [OHLCV(ts, 10.0 + index, 11.0 + index, 9.0 + index, 10.5 + index, 1.0)
               for index, ts in enumerate(timestamps)]
    _write_candles(path, candles, period)
    target = candles[target_index]

    with OHLCVReader(path) as reader:
        assert reader.get_positions(target.timestamp, target.timestamp) == (
            target_index,
            target_index + 1,
        )
        assert list(reader.read_from(target.timestamp, target.timestamp)) == [target]
        assert reader.read(target_index) == target


def __test_bisect_boundaries_on_irregular_series__(tmp_path: Path):
    """Inclusive bounds handle exact, between, outside, and reversed windows."""
    path = tmp_path / "bounds.ohlcv"
    candles = [
        OHLCV(1_000, 1.0, 2.0, 0.0, 1.5, 1.0),
        OHLCV(2_500, 2.0, 3.0, 1.0, 2.5, 1.0),
        OHLCV(10_000, 3.0, 4.0, 2.0, 3.5, 1.0),
    ]
    _write_candles(path, candles, "1S")

    with OHLCVReader(path) as reader:
        assert reader.get_positions(2_500, 2_500) == (1, 2)
        assert reader.get_positions(1_500, 9_000) == (1, 2)
        assert reader.get_positions(-1, 999) == (0, 0)
        assert reader.get_positions(10_001, 20_000) == (3, 3)
        assert reader.get_positions(9_000, 2_000) == (2, 2)
        assert reader.get_size(9_000, 2_000) == 0
        assert list(reader.read_from(9_000, 2_000)) == []
        assert list(reader.read_from(2_500)) == candles[1:]


def __test_single_record_bisect_needs_no_inferred_interval__(tmp_path: Path):
    """A one-record file supports timestamp addressing without a special case."""
    path = tmp_path / "single.ohlcv"
    candle = OHLCV(5_000, 1.0, 2.0, 0.0, 1.5, 1.0)
    _write_candles(path, [candle], "1M")

    with OHLCVReader(path) as reader:
        assert reader.get_positions() == (0, 1)
        assert reader.get_positions(5_000, 5_000) == (0, 1)
        assert reader.get_positions(4_999, 4_999) == (0, 0)
        assert reader.get_positions(5_001, 5_001) == (1, 1)
        assert list(reader.read_from(5_000)) == [candle]
        assert not reader.dense


def __test_empty_file_boundaries_and_datetimes__(tmp_path: Path):
    """An empty snapshot has stable zero bounds and no endpoint datetimes."""
    path = tmp_path / "empty.ohlcv"
    _write_candles(path, [], "01")

    with OHLCVReader(path) as reader:
        assert reader.period == "1"
        assert reader.size == 0
        assert reader.get_positions() == (0, 0)
        assert reader.get_positions(1, 2) == (0, 0)
        assert reader.get_size() == 0
        assert list(reader) == []
        assert list(reader.read_from(0)) == []
        assert reader.start_timestamp is None
        assert reader.end_timestamp is None
        with pytest.raises(AssertionError):
            _ = reader.start_datetime
        with pytest.raises(AssertionError):
            _ = reader.end_datetime


def __test_truncated_tail_is_ignored_and_recovered_on_append__(tmp_path: Path):
    """A partial uncommitted record is ignored by readers and removed by append-open."""
    path = tmp_path / "truncated.ohlcv"
    candles = [
        OHLCV(1_000, 1.0, 2.0, 0.0, 1.5, 1.0),
        OHLCV(2_000, 2.0, 3.0, 1.0, 2.5, 2.0),
    ]
    _write_candles(path, candles, "1S")
    committed_size = path.stat().st_size
    with path.open("ab") as file:
        file.write(b"power-cut-tail")

    with OHLCVReader(path) as reader:
        assert reader.size == 2
        assert list(reader) == candles
    assert path.stat().st_size == committed_size + len(b"power-cut-tail")

    appended = OHLCV(3_000, 3.0, 4.0, 2.0, 3.5, 3.0)
    with OHLCVWriter(path, "1S") as writer:
        assert path.stat().st_size == committed_size
        writer.write(appended)
    assert path.stat().st_size == committed_size + 36
    with OHLCVReader(path) as reader:
        assert list(reader) == candles + [appended]


def __test_append_updates_authoritative_header__(tmp_path: Path):
    """Reopening for append publishes the new count and actual last timestamp."""
    path = tmp_path / "append.ohlcv"
    first = OHLCV(1_000, 1.0, 2.0, 0.0, 1.5, 1.0)
    second = OHLCV(8_000, 2.0, 3.0, 1.0, 2.5, 2.0)
    _write_candles(path, [first], "1S")

    with OHLCVWriter(path, "1S") as writer:
        writer.write(second)
        assert writer.size == 2
        assert writer.start_timestamp == first.timestamp
        assert writer.end_timestamp == second.timestamp

    header = _header(path)
    assert header[7] == 2
    assert header[8] == first.timestamp
    assert header[9] == second.timestamp
    with OHLCVReader(path) as reader:
        assert list(reader) == [first, second]


@pytest.mark.parametrize(
    ("candle", "message"),
    [
        (OHLCV(999, 1.0, 2.0, 0.0, 1.5, 1.0), "strictly increasing"),
        (OHLCV(1_000, 1.0, 2.0, 0.0, 1.5, 1.0), "Duplicate OHLCV timestamp"),
        (OHLCV(2_000, 10.0, 9.0, 8.0, 10.0, 1.0), "high is below"),
        (OHLCV(2_000, 10.0, 11.0, 10.5, 10.0, 1.0), "low is above"),
    ],
)
def __test_invalid_append_is_rejected__(tmp_path: Path, candle: OHLCV, message: str):
    """Timestamp ordering and OHLC invariants reject malformed appends."""
    path = tmp_path / "reject.ohlcv"
    first = OHLCV(1_000, 1.0, 2.0, 0.0, 1.5, 1.0)
    _write_candles(path, [first], "1S")

    with OHLCVWriter(path, "1S") as writer:
        with pytest.raises(ValueError, match=message):
            writer.write(candle)
        assert writer.size == 1
    with OHLCVReader(path) as reader:
        assert list(reader) == [first]


def __test_dense_flag_is_verified_from_all_boundaries__(tmp_path: Path):
    """DENSE becomes true only for a complete fixed grid and never recovers after a hole."""
    dense_path = tmp_path / "dense.ohlcv"
    dense = [
        OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0),
        OHLCV(60_000, 2.0, 3.0, 1.0, 2.5, 1.0),
        OHLCV(120_000, 3.0, 4.0, 2.0, 3.5, 1.0),
    ]
    _write_candles(dense_path, dense)
    assert _header(dense_path)[6] == 1
    with OHLCVReader(dense_path) as reader:
        assert reader.dense

    sparse_path = tmp_path / "sparse.ohlcv"
    sparse = dense[:2] + [
        OHLCV(180_000, 3.0, 4.0, 2.0, 3.5, 1.0),
        OHLCV(240_000, 4.0, 5.0, 3.0, 4.5, 1.0),
    ]
    _write_candles(sparse_path, sparse)
    assert _header(sparse_path)[6] == 0
    with OHLCVReader(sparse_path) as reader:
        assert not reader.dense

    monthly_path = tmp_path / "monthly.ohlcv"
    _write_candles(monthly_path, dense[:2], "1M")
    with OHLCVReader(monthly_path) as reader:
        assert not reader.dense


def __test_f64_fallback_and_default_no_mintick_guard__(tmp_path: Path):
    """Economics-scale prices promote only when a mintick requests resolution checking."""
    candle = OHLCV(
        1_735_689_600_000,
        5_000_000_000_000.0,
        5_001_000_000_000.0,
        4_999_000_000_000.0,
        5_000_500_000_000.0,
        1.0,
    )
    promoted_path = tmp_path / "promoted.ohlcv"
    _write_candles(promoted_path, [candle], "1D", mintick=0.01)
    assert _record_size(promoted_path) == 48
    descriptors = [
        _DESCRIPTOR.unpack_from(promoted_path.read_bytes(), 64 + index * 24)
        for index in range(6)
    ]
    assert [(dtype, base, offset) for _, dtype, base, offset, _ in descriptors] == [
        (2, 255, 0),
        (6, 255, 8),
        (6, 255, 16),
        (6, 255, 24),
        (6, 255, 32),
        (6, 255, 40),
    ]
    assert promoted_path.read_bytes()[208:] == bytes.fromhex(
        "007c291f94010000000040e59c309242000068508b319242"
        "0000187aae2f92420000d41a14319242000000000000f03f"
    )
    with OHLCVReader(promoted_path) as reader:
        assert reader.read(0) == candle

    default_path = tmp_path / "default.ohlcv"
    _write_candles(default_path, [candle], "1D")
    assert _record_size(default_path) == 36


def __test_selective_promotion_uses_a_mixed_physical_layout__(tmp_path: Path):
    """Only failing price columns widen, preserving literal OHLCV field order."""
    path = tmp_path / "selective.ohlcv"
    candle = OHLCV(0, 0.0, 1_000_000_000.0, -0.001, 0.001, 1.0)
    _write_candles(path, [candle], mintick=0.01)

    raw = path.read_bytes()
    descriptors = [_DESCRIPTOR.unpack_from(raw, 64 + index * 24) for index in range(6)]
    assert _record_size(path) == 40
    assert [(dtype, base, offset) for _, dtype, base, offset, _ in descriptors] == [
        (2, 255, 0),
        (6, 255, 8),
        (6, 255, 16),
        (5, 1, 24),
        (5, 1, 28),
        (6, 255, 32),
    ]
    assert raw[208:] == bytes.fromhex(
        "000000000000000000000000000000000000000065cdcd41"
        "6f1283ba6f12833a000000000000f03f"
    )
    with OHLCVReader(path) as reader:
        _assert_candle_close(reader.read(0), candle, 0.01)


def __test_late_promotion_rebuilds_without_losing_data__(tmp_path: Path):
    """A reopened default file widens on a later failing delta without losing data."""
    path = tmp_path / "late.ohlcv"
    first = OHLCV(0, 5_000_000_000_000.0, 5_000_000_000_000.1,
                  4_999_999_999_999.9, 5_000_000_000_000.05, 1.0)
    second = OHLCV(60_000, 5_000_000_000_000.0, 5_001_000_000_000.0,
                   4_999_000_000_000.0, 5_000_500_000_000.0, 2.0)

    with OHLCVWriter(path, "1", mintick=0.01, truncate=True) as writer:
        writer.write(first)
        assert _record_size(path) == 36
    with OHLCVWriter(path, "1", mintick=0.01) as writer:
        writer.write(second)
        assert writer.size == 2
    assert _record_size(path) == 48
    assert list(tmp_path.glob(".late.ohlcv.*.tmp")) == []
    with OHLCVReader(path) as reader:
        assert reader.dense
        _assert_candle_close(reader.read(0), first, 0.01)
        assert reader.read(1) == second

    third = OHLCV(120_000, 10.0, 11.0, 9.0, 10.5, 3.0)
    with OHLCVWriter(path, "1") as writer:
        writer.write(third)
    assert _record_size(path) == 48


def __test_late_promotion_keeps_extra_fields__(tmp_path: Path):
    """A widening rebuild works while the sidecar holds the new bar's provisional row.

    The sidecar row of the bar being appended is written before the record is
    published, so during the rebuild it is deliberately one row ahead of the binary.
    The rebuild copies binary records only and must not validate that alignment.
    """
    path = tmp_path / "late_extra.ohlcv"
    first = OHLCV(0, 5_000_000_000_000.0, 5_000_000_000_000.1,
                  4_999_999_999_999.9, 5_000_000_000_000.05, 1.0,
                  extra_fields={"sig": 1.5})
    second = OHLCV(60_000, 5_000_000_000_000.0, 5_001_000_000_000.0,
                   4_999_000_000_000.0, 5_000_500_000_000.0, 2.0,
                   extra_fields={"sig": 2.5})

    with OHLCVWriter(path, "1", mintick=0.01, truncate=True) as writer:
        writer.write(first)
        assert _record_size(path) == 36
    with OHLCVWriter(path, "1", mintick=0.01) as writer:
        writer.write(second)
        assert writer.size == 2
    assert _record_size(path) == 48

    with OHLCVReader(path) as reader:
        assert reader.size == 2
        assert reader.read(0).extra_fields == {"sig": 1.5}
        assert reader.read(1).extra_fields == {"sig": 2.5}


@pytest.mark.parametrize("target_offset", [0, 64, _HEADER_SIZE, _HEADER_SIZE + 48])
def __test_promotion_retries_short_replacement_writes__(
    tmp_path: Path, monkeypatch, target_offset: int
):
    """Short replacement header, schema, and record writes resume at the correct offset."""
    path = tmp_path / f"replacement_short_{target_offset}.ohlcv"
    first = OHLCV(0, 5_000_000_000_000.0, 5_000_000_000_000.1,
                  4_999_999_999_999.9, 5_000_000_000_000.05, 1.0)
    second = OHLCV(60_000, 5_000_000_000_000.0, 5_001_000_000_000.0,
                   4_999_000_000_000.0, 5_000_500_000_000.0, 2.0)
    _write_candles(path, [first], mintick=0.01)
    real_named_temporary_file = ohlcv.tempfile.NamedTemporaryFile

    class ShortWriteProxy:
        def __init__(self, file):
            self._file = file
            self._shortened = False

        @property
        def name(self):
            return self._file.name

        def __enter__(self):
            self._file.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._file.__exit__(exc_type, exc_value, traceback)

        def write(self, data: bytes) -> int:
            if self._file.tell() == target_offset and not self._shortened:
                self._shortened = True
                return self._file.write(data[:-1])
            return self._file.write(data)

        def __getattr__(self, name):
            return getattr(self._file, name)

    def short_named_temporary_file(*args, **kwargs):
        return ShortWriteProxy(real_named_temporary_file(*args, **kwargs))

    monkeypatch.setattr(ohlcv.tempfile, "NamedTemporaryFile", short_named_temporary_file)
    with OHLCVWriter(path, "1", mintick=0.01) as writer:
        writer.write(second)
        assert writer.size == 2

    assert _record_size(path) == 48
    with OHLCVReader(path) as reader:
        assert reader.size == 2
        assert reader.read(1) == second


def __test_post_replace_failure_keeps_writer_on_current_inode__(tmp_path: Path, monkeypatch):
    """A directory fsync error leaves subsequent writes attached to the replaced path."""
    path = tmp_path / "post_replace_failure.ohlcv"
    first = OHLCV(0, 5_000_000_000_000.0, 5_000_000_000_000.1,
                  4_999_999_999_999.9, 5_000_000_000_000.05, 1.0)
    second = OHLCV(60_000, 5_000_000_000_000.0, 5_001_000_000_000.0,
                   4_999_000_000_000.0, 5_000_500_000_000.0, 2.0)
    third = OHLCV(120_000, 10.0, 11.0, 9.0, 10.5, 3.0)
    writer = OHLCVWriter(path, "1", mintick=0.01, truncate=True).open()
    writer.write(first)

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError(f"injected directory fsync failure: {directory}")

    monkeypatch.setattr(ohlcv, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        writer.write(second)
    assert writer.size == 2
    writer.write(third)
    writer.close()

    with OHLCVReader(path) as reader:
        assert reader.size == 3
        assert reader.read(1) == second
        assert reader.read(2) == third


def __test_post_replace_failure_keeps_committed_extra_row__(tmp_path: Path, monkeypatch):
    """A post-publication rebuild error keeps the extra fields of the committed bar."""
    path = tmp_path / "post_replace_failure_extra.ohlcv"
    first = OHLCV(0, 5_000_000_000_000.0, 5_000_000_000_000.1,
                  4_999_999_999_999.9, 5_000_000_000_000.05, 1.0,
                  extra_fields={"tag": "one"})
    second = OHLCV(60_000, 5_000_000_000_000.0, 5_001_000_000_000.0,
                   4_999_000_000_000.0, 5_000_500_000_000.0, 2.0,
                   extra_fields={"tag": "two"})
    third = OHLCV(120_000, 10.0, 11.0, 9.0, 10.5, 3.0, extra_fields={"tag": "three"})
    writer = OHLCVWriter(path, "1", mintick=0.01, truncate=True).open()
    writer.write(first)

    def fail_directory_fsync(directory: Path) -> None:
        raise OSError(f"injected directory fsync failure: {directory}")

    monkeypatch.setattr(ohlcv, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        writer.write(second)
    assert writer.size == 2
    monkeypatch.undo()
    writer.write(third)
    writer.close()

    with OHLCVReader(path) as reader:
        assert reader.size == 3
        assert reader.read(0).extra_fields == {"tag": "one"}
        assert reader.read(1).extra_fields == {"tag": "two"}
        assert reader.read(2).extra_fields == {"tag": "three"}


def __test_period_and_mintick_validation__(tmp_path: Path):
    """Periods canonicalize strictly and mintick must be finite and positive."""
    assert OHLCVWriter(tmp_path / "minutes", "001").period == "1"
    assert OHLCVWriter(tmp_path / "days", "01D").period == "1D"
    for period in ("", "0", "0D", "1H", "d", "D", "-1", "4294967296"):
        with pytest.raises(ValueError):
            OHLCVWriter(tmp_path / "invalid", period)
    for mintick in (0.0, -1.0, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            OHLCVWriter(tmp_path / "invalid_tick", "1", mintick=mintick)


def __test_hour_interval_is_read_as_minutes__(tmp_path: Path):
    """A valid unit-3 fixture canonicalizes hours to TradingView minute notation."""
    columns = ohlcv._make_default_columns()
    raw_columns = tuple(
        _raw_descriptor(column.role, column.dtype, column.base, column.byte_offset, column.name)
        for column in columns
    )
    record = struct.pack("<qdfffd", 1_000, 1.0, 1.0, -1.0, 0.5, 2.0)
    path = tmp_path / "hours.ohlcv"
    _write_raw_file(path, raw_columns, [record], interval_value=4, interval_unit=3)

    with OHLCVReader(path) as reader:
        assert reader.period == "240"


def __test_writer_state_machine_and_one_shot_truncate__(tmp_path: Path):
    """Closed operations fail, open/close are idempotent, and truncate-on-open is one-shot."""
    path = tmp_path / "state.ohlcv"
    path.write_bytes(b"not an OHLCV file")
    writer = OHLCVWriter(path, "1", truncate=True)
    with pytest.raises(RuntimeError, match="not open"):
        writer.write(OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0))
    with pytest.raises(RuntimeError, match="not open"):
        writer.truncate()

    assert writer.open() is writer
    assert writer.open() is writer
    writer.write(OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0))
    writer.close()
    writer.close()
    assert writer.open() is writer
    assert writer.size == 1
    writer.truncate()
    assert writer.size == 0
    writer.close()

    with OHLCVReader(path) as reader:
        assert reader.size == 0


def __test_timestamp_i64_boundaries__(tmp_path: Path):
    """Signed i64 endpoints are accepted and values outside the range are rejected."""
    low_path = tmp_path / "low.ohlcv"
    _write_candles(low_path, [OHLCV(-(1 << 63), 1.0, 2.0, 0.0, 1.5, 1.0)], "1S")
    with OHLCVReader(low_path) as reader:
        assert reader.start_timestamp == -(1 << 63)

    high_path = tmp_path / "high.ohlcv"
    _write_candles(high_path, [OHLCV((1 << 63) - 1, 1.0, 2.0, 0.0, 1.5, 1.0)], "1S")
    with OHLCVReader(high_path) as reader:
        assert reader.end_timestamp == (1 << 63) - 1

    for timestamp in (-(1 << 63) - 1, 1 << 63):
        path = tmp_path / f"outside_{timestamp}.ohlcv"
        with OHLCVWriter(path, "1S", truncate=True) as writer:
            with pytest.raises(ValueError, match="signed i64"):
                writer.write(OHLCV(timestamp, 1.0, 2.0, 0.0, 1.5, 1.0))


def __test_nan_encoding_and_infinity_rejection__(tmp_path: Path):
    """NaNs use canonical bit patterns and infinities are rejected for every value field."""
    path = tmp_path / "nan.ohlcv"
    candle = OHLCV(1_000, math.nan, math.nan, math.nan, math.nan, math.nan)
    _write_candles(path, [candle])
    record = path.read_bytes()[208:244]
    assert record[8:16] == bytes.fromhex("000000000000f87f")
    assert record[16:20] == bytes.fromhex("0000c07f")
    assert record[20:24] == bytes.fromhex("0000c07f")
    assert record[24:28] == bytes.fromhex("0000c07f")
    assert record[28:36] == bytes.fromhex("000000000000f87f")
    with OHLCVReader(path) as reader:
        actual = reader.read(0)
        assert all(math.isnan(value) for value in actual[1:6])

    finite = [1.0, 2.0, 0.0, 1.5, 1.0]
    for index in range(5):
        values = finite.copy()
        values[index] = math.inf if index % 2 == 0 else -math.inf
        invalid = OHLCV(2_000, *values)
        with OHLCVWriter(tmp_path / f"inf_{index}.ohlcv", "1", truncate=True) as writer:
            with pytest.raises(ValueError, match="infinity"):
                writer.write(invalid)


def __test_nan_gates_ohlc_checks_and_negative_volume_is_allowed__(tmp_path: Path):
    """NaN price relations are omitted and negative finite volume remains representable."""
    path = tmp_path / "nan_gate.ohlcv"
    candles = [
        OHLCV(1_000, math.nan, math.nan, math.nan, math.nan, -1.0),
        OHLCV(2_000, 1.0, math.nan, 0.0, 2.0, -2.0),
        OHLCV(3_000, 1.0, -100.0, math.nan, math.nan, -3.0),
    ]
    _write_candles(path, candles)
    with OHLCVReader(path) as reader:
        assert reader.size == 3
        assert [item.volume for item in reader] == [-1.0, -2.0, -3.0]


@pytest.mark.parametrize(
    ("name", "candle"),
    [
        ("nan_base", OHLCV(0, math.nan, 12.0, 8.0, 11.0, 7.0)),
        ("overflowing_subtraction", OHLCV(0, 1e308, 1.7e308, -1e308, 0.0, 1.0)),
        ("f32_overflow", OHLCV(0, 0.0, 1e308, -1e308, 0.0, 1.0)),
    ],
)
def __test_unrepresentable_default_deltas_are_rejected__(
    tmp_path: Path, name: str, candle: OHLCV
):
    """Finite prices must not become NaN, infinity, or overflowed f32 deltas."""
    path = tmp_path / f"unrepresentable_{name}.ohlcv"
    with OHLCVWriter(path, "1", truncate=True) as writer:
        with pytest.raises(ValueError, match="delta|storage range"):
            writer.write(candle)
        assert writer.size == 0
    with OHLCVReader(path) as reader:
        assert reader.size == 0


def __test_toml_sidecar_does_not_affect_layout__(tmp_path: Path):
    """A neighboring TOML cannot supply mintick or otherwise alter the binary schema."""
    path = tmp_path / "sidecar.ohlcv"
    path.with_suffix(".toml").write_text("mintick = 0.01\nperiod = \"1D\"\n", encoding="utf-8")
    candle = OHLCV(0, 5e12, 5.001e12, 4.999e12, 5.0005e12, 1.0)
    _write_candles(path, [candle], "1")

    assert _record_size(path) == 36
    with OHLCVReader(path) as reader:
        assert reader.period == "1"


def __test_reader_snapshot_ignores_tail_and_later_growth__(tmp_path: Path):
    """An open reader maps only its original committed extent and never follows growth."""
    path = tmp_path / "snapshot.ohlcv"
    first = OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)
    second = OHLCV(60_000, 2.0, 3.0, 1.0, 2.5, 2.0)
    _write_candles(path, [first])
    with path.open("ab") as file:
        file.write(b"ignored-tail")

    with OHLCVReader(path) as snapshot:
        assert snapshot.size == 1
        with OHLCVWriter(path, "1") as writer:
            writer.write(second)
        assert snapshot.size == 1
        assert list(snapshot) == [first]

    with OHLCVReader(path) as current:
        assert current.size == 2
        assert list(current) == [first, second]


def __test_read_error_priority_and_idempotent_reader_state__(tmp_path: Path):
    """Range checking precedes open-state checking and reader open/close are idempotent."""
    path = tmp_path / "reader_state.ohlcv"
    candle = OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)
    _write_candles(path, [candle])
    reader = OHLCVReader(path)

    with pytest.raises(IndexError):
        reader.read(-1)
    with pytest.raises(IndexError):
        reader.read(1)
    reader.open()
    reader.close()
    with pytest.raises(RuntimeError, match="not open"):
        reader.read(0)
    assert reader.open() is reader
    assert reader.open() is reader
    assert reader.read(0) == candle
    reader.close()
    reader.close()


def __test_datetime_properties_and_exports__(tmp_path: Path):
    """Endpoint datetimes and CSV/JSON exports use UTC millisecond timestamps."""
    path = tmp_path / "export.ohlcv"
    candle = OHLCV(1_735_689_600_123, 1.0, 2.0, 0.0, 1.5, math.nan)
    _write_candles(path, [candle])
    csv_path = tmp_path / "out.csv"
    csv_datetime_path = tmp_path / "out_datetime.csv"
    json_path = tmp_path / "out.json"
    json_datetime_path = tmp_path / "out_datetime.json"

    with OHLCVReader(path) as reader:
        expected_datetime = datetime(2025, 1, 1, 0, 0, 0, 123_000, tzinfo=UTC)
        assert reader.start_datetime == expected_datetime
        assert reader.end_datetime == expected_datetime
        reader.save_to_csv(csv_path)
        reader.save_to_csv(csv_datetime_path, as_datetime=True)
        reader.save_to_json(json_path)
        reader.save_to_json(json_datetime_path, as_datetime=True)

    assert csv_path.read_text().splitlines()[0] == "timestamp,open,high,low,close,volume"
    assert csv_path.read_text().splitlines()[1].startswith("1735689600123,")
    assert csv_datetime_path.read_text().splitlines()[0] == "time,open,high,low,close,volume"
    assert csv_datetime_path.read_text().splitlines()[1].startswith("2025-01-01 00:00:00.123000+00:00,")
    assert json.loads(json_path.read_text())[0]["timestamp"] == candle.timestamp
    datetime_entry = json.loads(json_datetime_path.read_text())[0]
    assert datetime_entry["time"] == "2025-01-01T00:00:00.123000+00:00"
    assert "timestamp" not in datetime_entry
    assert "NaN" in json_path.read_text()


def __test_generic_reader_supports_delta_chains_and_extra_columns__(tmp_path: Path):
    """The generic decoder follows acyclic delta chains while the append writer stays strict."""
    columns = (
        _raw_descriptor(0, 2, 255, 0, "timestamp"),
        _raw_descriptor(1, 6, 255, 8, "open"),
        _raw_descriptor(4, 5, 1, 16, "close"),
        _raw_descriptor(2, 5, 4, 20, "high"),
        _raw_descriptor(3, 5, 1, 24, "low"),
        _raw_descriptor(5, 6, 255, 28, "volume"),
        _raw_descriptor(255, 6, 255, 36, "spread"),
    )
    record = struct.pack("<qdfffdd", 1_000, 100.0, 0.5, 1.5, -1.0, 10.0, 0.25)
    path = tmp_path / "generic.ohlcv"
    _write_raw_file(path, columns, [record])

    with OHLCVReader(path) as reader:
        assert reader.read(0) == OHLCV(1_000, 100.0, 102.0, 99.0, 100.5, 10.0)
    with pytest.raises(ValueError, match="standard OHLCV profile"):
        OHLCVWriter(path, "1").open()


def __test_schema_validation_rules__():
    """Column roles, names, bases, timestamp, and packed ranges reject malformed schemas."""
    column = ohlcv._Column
    valid = list(ohlcv._make_default_columns())

    with pytest.raises(ValueError, match="column role"):
        ohlcv._validate_columns((column(14, 2, 255, 0, "bad"),), 8)
    invalid_dtype = valid.copy()
    invalid_dtype[1] = column(1, 3, 255, 8, "open")
    with pytest.raises(ValueError, match="column dtype"):
        ohlcv._validate_columns(tuple(invalid_dtype), 36)
    invalid_base = valid.copy()
    invalid_base[2] = column(2, 5, 14, 16, "high")
    with pytest.raises(ValueError, match="delta base"):
        ohlcv._validate_columns(tuple(invalid_base), 36)
    duplicate_role = valid.copy()
    duplicate_role[1] = column(0, 6, 255, 8, "open")
    with pytest.raises(ValueError, match="Duplicate OHLCV column role"):
        ohlcv._validate_columns(tuple(duplicate_role), 36)

    custom_columns = tuple(valid) + (
        column(255, 6, 255, 36, "extra"),
        column(255, 6, 255, 44, "extra"),
    )
    with pytest.raises(ValueError, match="Duplicate custom"):
        ohlcv._validate_columns(custom_columns, 52)

    missing_base = valid.copy()
    missing_base[2] = column(2, 5, 6, 16, "high")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        ohlcv._validate_columns(tuple(missing_base), 36)
    cycle = valid.copy()
    cycle[1] = column(1, 6, 2, 8, "open")
    cycle[2] = column(2, 5, 1, 16, "high")
    with pytest.raises(ValueError, match="cycle"):
        ohlcv._validate_columns(tuple(cycle), 36)

    missing_timestamp = tuple(valid[1:])
    with pytest.raises(ValueError, match="exactly one timestamp"):
        ohlcv._validate_columns(missing_timestamp, 28)
    bad_timestamp = valid.copy()
    bad_timestamp[0] = column(0, 6, 255, 0, "timestamp")
    with pytest.raises(ValueError, match="absolute i64"):
        ohlcv._validate_columns(tuple(bad_timestamp), 36)
    gap = valid.copy()
    gap[1] = column(1, 6, 255, 9, "open")
    with pytest.raises(ValueError, match="without gaps"):
        ohlcv._validate_columns(tuple(gap), 36)
    with pytest.raises(ValueError, match="do not match record_size"):
        ohlcv._validate_columns(tuple(valid), 40)


def __test_column_name_and_descriptor_validation__():
    """Column names require unique nonempty ASCII bytes with canonical NUL padding."""
    assert ohlcv._column_name(b"close" + b"\0" * 13) == "close"
    for raw_name, message in (
        (b"\0" * 18, "cannot be empty"),
        (b"x\0y" + b"\0" * 15, "NUL padding"),
        (b"\xff" + b"\0" * 17, "ASCII"),
    ):
        with pytest.raises(ValueError, match=message):
            ohlcv._column_name(raw_name)

    for name in ("", "a" * 19, "a\0b"):
        with pytest.raises(ValueError, match="Invalid OHLCV column name"):
            ohlcv._pack_descriptor(ohlcv._Column(255, 6, 255, 0, name))
    with pytest.raises(ValueError, match="Invalid OHLCV column name"):
        ohlcv._pack_descriptor(ohlcv._Column(255, 6, 255, 0, "ár"))


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, b"BADMAGIC", "Invalid OHLCV v2 magic"),
        (8, struct.pack("<H", 3), "major version"),
        (10, struct.pack("<H", 1), "minor version"),
        (22, struct.pack("<H", 2), "header flags"),
        (20, struct.pack("<H", 0), "schema cannot be empty"),
        (12, struct.pack("<I", 64), "header_size"),
        (16, struct.pack("<I", 0), "record_size"),
        (48, struct.pack("<I", 0), "interval_value"),
        (52, b"\x07", "interval unit"),
        (24, struct.pack("<Q", 2), "missing data committed"),
        (32, struct.pack("<q", 999), "header timestamps"),
        (40, struct.pack("<q", 999), "header timestamps"),
    ],
)
def __test_corrupt_headers_are_rejected__(
    tmp_path: Path, offset: int, replacement: bytes, message: str
):
    """Independent header invariants reject corrupt files before mmap exposure."""
    path = tmp_path / f"corrupt_{offset}_{replacement.hex()}.ohlcv"
    _write_candles(path, [OHLCV(1_000, 1.0, 2.0, 0.0, 1.5, 1.0)])
    _mutate(path, offset, replacement)

    with pytest.raises(ValueError, match=message):
        ohlcv._V2OHLCVReader(path).open()


def __test_header_length_and_dense_edge_corruption__(tmp_path: Path):
    """Short headers, oversized descriptor declarations, and impossible DENSE states fail."""
    short_path = tmp_path / "short.ohlcv"
    short_path.write_bytes(_MAGIC)
    with pytest.raises(ValueError, match="too short"):
        OHLCVReader(short_path).open()

    declared_path = tmp_path / "declared.ohlcv"
    _write_candles(declared_path, [])
    _mutate(declared_path, 12, struct.pack("<I", 232))
    _mutate(declared_path, 20, struct.pack("<H", 7))
    with pytest.raises(ValueError, match="shorter than its declared header"):
        OHLCVReader(declared_path).open()

    empty_dense = tmp_path / "empty_dense.ohlcv"
    _write_candles(empty_dense, [])
    _mutate(empty_dense, 22, struct.pack("<H", 1))
    with pytest.raises(ValueError, match="Empty OHLCV files cannot be marked DENSE"):
        OHLCVReader(empty_dense).open()

    empty_timestamp = tmp_path / "empty_timestamp.ohlcv"
    _write_candles(empty_timestamp, [])
    _mutate(empty_timestamp, 32, struct.pack("<q", 1))
    with pytest.raises(ValueError, match="zero first and last"):
        OHLCVReader(empty_timestamp).open()

    one_dense = tmp_path / "one_dense.ohlcv"
    _write_candles(one_dense, [OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)])
    _mutate(one_dense, 22, struct.pack("<H", 1))
    with pytest.raises(ValueError, match="one-record"):
        OHLCVReader(one_dense).open()


def __test_corrupt_descriptor_names_are_rejected__(tmp_path: Path):
    """Descriptor names reject empty, non-ASCII, and nonzero bytes after the first NUL."""
    cases = [
        (b"\0" * 18, "cannot be empty"),
        (b"\xff" + b"\0" * 17, "ASCII"),
        (b"ts\0x" + b"\0" * 14, "NUL padding"),
    ]
    for index, (raw_name, message) in enumerate(cases):
        path = tmp_path / f"descriptor_name_{index}.ohlcv"
        _write_candles(path, [])
        _mutate(path, 64 + 6, raw_name)
        with pytest.raises(ValueError, match=message):
            OHLCVReader(path).open()


def __test_failed_initial_writer_open_closes_and_can_retry__(tmp_path: Path, monkeypatch):
    """A failed empty-file fsync releases the handle and preserves create-on-retry state."""
    path = tmp_path / "writer_retry.ohlcv"
    writer = OHLCVWriter(path, "1")
    real_fsync = os.fsync
    failed = False

    def fail_once(fd: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(ohlcv.os, "fsync", fail_once)
    with pytest.raises(OSError, match="injected fsync failure"):
        writer.open()
    assert writer._file is None

    assert writer.open() is writer
    writer.close()
    with OHLCVReader(path) as reader:
        assert reader.size == 0


def __test_failed_reader_open_can_be_retried__(tmp_path: Path):
    """A validation failure closes the handle so the same reader can open a repaired file."""
    path = tmp_path / "retry.ohlcv"
    path.write_bytes(b"broken")
    reader = OHLCVReader(path)
    with pytest.raises(ValueError):
        reader.open()

    candle = OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)
    _write_candles(path, [candle])
    assert reader.open() is reader
    assert reader.read(0) == candle
    reader.close()


def __test_period_mismatch_rejects_append_without_modifying_file__(tmp_path: Path):
    """Append requires the exact canonical period declared by the file header."""
    path = tmp_path / "period_mismatch.ohlcv"
    _write_candles(path, [OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)], "1")
    before = path.read_bytes()

    writer = OHLCVWriter(path, "5")
    with pytest.raises(ValueError, match="period mismatch"):
        writer.open()
    writer.close()
    assert path.read_bytes() == before


def __test_record_and_header_publication_order__(tmp_path: Path, monkeypatch):
    """A record fsync completes before the authoritative header count is published."""
    path = tmp_path / "order.ohlcv"
    writer = OHLCVWriter(path, "1", truncate=True).open()
    events: list[str] = []
    real_pwrite = os.pwrite
    real_fsync = os.fsync

    def tracking_pwrite(fd: int, data: bytes, offset: int) -> int:
        events.append("record" if offset == _HEADER_SIZE else "header")
        return real_pwrite(fd, data, offset)

    def tracking_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    monkeypatch.setattr(ohlcv.os, "pwrite", tracking_pwrite)
    monkeypatch.setattr(ohlcv.os, "fsync", tracking_fsync)
    writer.write(OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0))
    assert events == ["record", "fsync", "header", "fsync"]
    writer.close()


def __test_short_record_write_is_never_published__(tmp_path: Path, monkeypatch):
    """A short record pwrite leaves both in-memory and on-disk record counts unchanged."""
    path = tmp_path / "short_record.ohlcv"
    writer = OHLCVWriter(path, "1", truncate=True).open()
    real_pwrite = os.pwrite

    def short_record(fd: int, data: bytes, offset: int) -> int:
        if offset == _HEADER_SIZE:
            return len(data) - 1
        return real_pwrite(fd, data, offset)

    monkeypatch.setattr(ohlcv.os, "pwrite", short_record)
    with pytest.raises(OSError, match="complete OHLCV record"):
        writer.write(OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0))
    assert writer.size == 0
    writer.close()
    with OHLCVReader(path) as reader:
        assert reader.size == 0


def __test_short_header_write_leaves_record_as_uncommitted_tail__(tmp_path: Path, monkeypatch):
    """A short header pwrite cannot expose the already durable record bytes."""
    path = tmp_path / "short_header.ohlcv"
    writer = OHLCVWriter(path, "1", truncate=True).open()
    real_pwrite = os.pwrite

    shortened = False

    def short_header(fd: int, data: bytes, offset: int) -> int:
        nonlocal shortened
        if offset == 0 and len(data) == 64 and not shortened:
            shortened = True
            return real_pwrite(fd, data[:-1], offset)
        return real_pwrite(fd, data, offset)

    monkeypatch.setattr(ohlcv.os, "pwrite", short_header)
    with pytest.raises(OSError, match="complete OHLCV header"):
        writer.write(OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0))
    assert writer.size == 0
    writer.close()
    assert path.stat().st_size == _HEADER_SIZE + 36
    with OHLCVReader(path) as reader:
        assert reader.size == 0


def __test_partial_header_rollback_preserves_the_previous_commit__(tmp_path: Path, monkeypatch):
    """A real 32-byte header prefix is rolled back to the prior committed metadata."""
    path = tmp_path / "partial_header_rollback.ohlcv"
    first = OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)
    second = OHLCV(60_000, 2.0, 3.0, 1.0, 2.5, 2.0)
    _write_candles(path, [first])
    writer = OHLCVWriter(path, "1").open()
    real_pwrite = os.pwrite
    shortened = False

    def short_header(fd: int, data: bytes, offset: int) -> int:
        nonlocal shortened
        if offset == 0 and len(data) == 64 and not shortened:
            shortened = True
            return real_pwrite(fd, data[:32], offset)
        return real_pwrite(fd, data, offset)

    monkeypatch.setattr(ohlcv.os, "pwrite", short_header)
    with pytest.raises(OSError, match="complete OHLCV header"):
        writer.write(second)
    assert writer.size == 1
    writer.close()
    with OHLCVReader(path) as reader:
        assert list(reader) == [first]


def __test_f32_resolution_and_promotion_boundaries__(tmp_path: Path):
    """Resolution measurement covers deltas, signed values, overflow, NaN, and equality."""
    assert ohlcv._f32_resolution(0.0) == 2.0**-149
    assert ohlcv._f32_resolution(-0.0) == 2.0**-149
    assert ohlcv._f32_resolution(1.0) == 2.0**-23
    assert ohlcv._f32_resolution(-1.0) == 2.0**-23
    assert ohlcv._f32_resolution(math.nan) is None
    assert ohlcv._f32_resolution(math.inf) is None
    assert ohlcv._f32_resolution(1e100) is None

    exactly_half_tick = OHLCV(0, 0.0, 1.0, -1.0, 1.0, 1.0)
    resolution = ohlcv._f32_resolution(1.0)
    assert resolution is not None
    roles = ohlcv._failing_delta_roles(exactly_half_tick, resolution * 2)
    assert roles == frozenset({2, 3, 4})
    assert ohlcv._failing_delta_roles(exactly_half_tick, None) == frozenset()

    stable_deltas = OHLCV(
        0,
        100_000.0,
        100_000.01,
        99_999.99,
        100_000.005,
        1.0,
    )
    assert ohlcv._failing_delta_roles(stable_deltas, 0.01) == frozenset()
    stable_path = tmp_path / "stable_deltas.ohlcv"
    _write_candles(stable_path, [stable_deltas], mintick=0.01)
    assert _record_size(stable_path) == 36

    nan_target = OHLCV(0, 1.0, math.nan, 0.0, 1.0, 1.0)
    assert 2 not in ohlcv._failing_delta_roles(nan_target, 0.01)
    nan_open = OHLCV(0, math.nan, 2.0, math.nan, 1.0, 1.0)
    assert ohlcv._failing_delta_roles(nan_open, 0.01) == frozenset({2, 4})


def __test_public_reader_transparently_reads_legacy_v1__(tmp_path: Path):
    """The public reader selects v1 once and exposes millisecond timestamps."""
    path = tmp_path / "legacy.ohlcv"
    first_seconds = 1_700_000_000
    records = (
        struct.pack("Ifffff", first_seconds, 1.0, 2.0, 0.5, 1.5, 10.0),
        struct.pack("Ifffff", first_seconds + 60, 1.5, 2.5, 1.0, 2.0, 20.0),
    )
    path.write_bytes(b"".join(records))

    with OHLCVReader(path) as reader:
        assert reader.period is None
        assert reader.dense is None
        assert reader.size == 2
        assert reader.start_timestamp == first_seconds * 1000
        assert reader.end_timestamp == (first_seconds + 60) * 1000
        assert [candle.timestamp for candle in reader] == [
            first_seconds * 1000,
            (first_seconds + 60) * 1000,
        ]
        assert [candle.close for candle in reader.read_from(first_seconds * 1000)] == [1.5, 2.0]


def __test_writer_keeps_extra_sidecar_position_aligned__(tmp_path: Path):
    """Extra fields round-trip through a sidecar holding one row per committed record."""
    path = tmp_path / "extra.ohlcv"
    sidecar = path.with_suffix(".extra.csv")

    # The first two bars carry nothing, so they must be back-padded once the third
    # bar introduces the columns; otherwise row N would describe the wrong record.
    candles = [
        OHLCV(60_000 * index, 1.0 + index, 2.0 + index, 0.5, 1.5 + index, 10.0,
              extra_fields={"ask": 1.25 + index, "note": f"n{index}"} if index >= 2 else None)
        for index in range(4)
    ]
    _write_candles(path, candles)

    assert sidecar.read_text().splitlines() == [
        "ask,note", ",", ",", "3.25,n2", "4.25,n3",
    ]

    appended = OHLCV(240_000, 5.0, 6.0, 4.0, 5.5, 10.0,
                     extra_fields={"ask": 9.5, "note": "n4"})
    with OHLCVWriter(path, "1") as writer:
        writer.write(appended)

    with OHLCVReader(path) as reader:
        assert reader.size == 5
        assert reader.read(3).extra_fields == {"ask": 4.25, "note": "n3"}
        assert reader.read(4).extra_fields == {"ask": 9.5, "note": "n4"}
        empty = reader.read(0).extra_fields
        assert empty is not None and math.isnan(empty["ask"]) and empty["note"] == ""


def __test_failed_sidecar_write_commits_nothing__(tmp_path: Path):
    """A bar whose sidecar row cannot be written stays uncommitted and retryable."""
    path = tmp_path / "sidecar_fail.ohlcv"
    candles = [
        OHLCV(60_000 * index, 1.0, 2.0, 0.5, 1.5, 10.0, extra_fields={"sig": float(index)})
        for index in range(3)
    ]

    with OHLCVWriter(path, "1", truncate=True) as writer:
        writer.write(candles[0])
        writer.write(candles[1])
        # Break the open sidecar so the next row write fails after the record bytes
        # have already been laid down.
        assert writer._extra_file is not None
        writer._extra_file.close()
        with pytest.raises(ValueError):
            writer.write(candles[2])
        assert writer.size == 2
        # The bar was never published, so the very same one can be written again.
        writer.write(candles[2])

    with OHLCVReader(path) as reader:
        assert reader.size == 3
        assert [candle.extra_fields for candle in reader] == [
            {"sig": 0.0}, {"sig": 1.0}, {"sig": 2.0},
        ]


def __test_writer_truncate_drops_the_extra_sidecar__(tmp_path: Path):
    """A sidecar cannot outlive the records it describes, in either truncation path."""
    path = tmp_path / "stale.ohlcv"
    sidecar = path.with_suffix(".extra.csv")
    candle = OHLCV(0, 1.0, 2.0, 0.5, 1.5, 10.0, extra_fields={"ask": 1.25})
    _write_candles(path, [candle])
    assert sidecar.exists()

    with OHLCVWriter(path, "1") as writer:
        writer.truncate()
    assert not sidecar.exists()

    _write_candles(path, [candle])
    assert sidecar.exists()
    OHLCVWriter(path, "1", truncate=True).open().close()
    assert not sidecar.exists()

    # A stale sidecar would otherwise make the reader reject the pair outright.
    with OHLCVReader(path) as reader:
        assert reader.size == 0


def __test_writer_rejects_appending_to_legacy_v1_file__(tmp_path: Path):
    """Appending to v1 is refused and leaves the file untouched."""
    path = tmp_path / "legacy_write.ohlcv"
    legacy_data = struct.pack("Ifffff", 1_700_000_000, 1.0, 2.0, 0.5, 1.5, 10.0)
    path.write_bytes(legacy_data)

    with pytest.raises(ValueError, match="legacy v1.*re-download"):
        OHLCVWriter(path, "1").open()
    assert path.read_bytes() == legacy_data


def __test_writer_truncation_replaces_legacy_v1_file__(tmp_path: Path):
    """Truncation replaces any existing file, which is how v1 data is re-downloaded."""
    path = tmp_path / "legacy_replace.ohlcv"
    path.write_bytes(struct.pack("Ifffff", 1_700_000_000, 1.0, 2.0, 0.5, 1.5, 10.0))

    candle = OHLCV(1_700_000_000_000, 1.0, 2.0, 0.5, 1.5, 10.0)
    with OHLCVWriter(path, "1", truncate=True) as writer:
        writer.write(candle)

    with OHLCVReader(path) as reader:
        assert reader.period == "1"
        assert list(reader) == [candle]


def __test_record_count_matches_v2_reader_size__(tmp_path: Path):
    """A v2 probe reports the reader's committed count without opening a reader."""
    path = tmp_path / "count_v2.ohlcv"
    candles = [
        OHLCV(60_000 * index, 1.0 + index, 2.0 + index, float(index), 1.5 + index, 1.0)
        for index in range(5)
    ]
    _write_candles(path, candles)

    with OHLCVReader(path) as reader:
        assert record_count(path) == reader.size == 5

    empty_path = tmp_path / "count_v2_empty.ohlcv"
    _write_candles(empty_path, [])
    with OHLCVReader(empty_path) as reader:
        assert record_count(empty_path) == reader.size == 0


def __test_record_count_matches_legacy_v1_reader_size__(tmp_path: Path):
    """A legacy v1 probe reports the reader's count derived from the v1 record size."""
    path = tmp_path / "count_v1.ohlcv"
    first_seconds = 1_700_000_000
    path.write_bytes(
        b"".join(
            struct.pack("Ifffff", first_seconds + index * 60, 1.0, 2.0, 0.5, 1.5, 10.0)
            for index in range(3)
        )
    )

    with OHLCVReader(path) as reader:
        assert record_count(path) == reader.size == 3


def __test_record_count_returns_zero_for_unusable_files__(tmp_path: Path):
    """Absent, empty, truncated, corrupt, and non-record files all probe as zero."""
    assert record_count(tmp_path / "absent.ohlcv") == 0

    empty_path = tmp_path / "empty_bytes.ohlcv"
    empty_path.write_bytes(b"")
    assert record_count(empty_path) == 0

    magic_only_path = tmp_path / "magic_only.ohlcv"
    magic_only_path.write_bytes(_MAGIC)
    assert record_count(magic_only_path) == 0

    corrupt_path = tmp_path / "corrupt_version.ohlcv"
    _write_candles(corrupt_path, [OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0)])
    _mutate(corrupt_path, 8, struct.pack("<H", 3))
    assert record_count(corrupt_path) == 0

    csv_path = tmp_path / "actually_a_csv.ohlcv"
    csv_path.write_text(
        "time,open,high,low,close,volume\n2025-01-01T00:00:00Z,1,2,0.5,1.5,100\n",
        encoding="utf-8",
    )
    # Not a whole multiple of the legacy v1 record stride, so it cannot be read as one.
    assert csv_path.stat().st_size % ohlcv._LEGACY_RECORD_SIZE != 0
    assert record_count(csv_path) == 0


def __test_record_count_ignores_uncommitted_v2_tail__(tmp_path: Path):
    """The header count wins over the byte-derived count on a partially written file."""
    path = tmp_path / "count_tail.ohlcv"
    candles = [
        OHLCV(0, 1.0, 2.0, 0.0, 1.5, 1.0),
        OHLCV(60_000, 2.0, 3.0, 1.0, 2.5, 2.0),
    ]
    _write_candles(path, candles)
    committed_size = path.stat().st_size
    with path.open("ab") as file:
        file.write(struct.pack("<qdfffd", 120_000, 3.0, 1.0, -1.0, 0.5, 3.0))

    assert path.stat().st_size == committed_size + 36
    assert (path.stat().st_size - _HEADER_SIZE) // 36 == 3
    assert record_count(path) == 2
    with OHLCVReader(path) as reader:
        assert reader.size == 2
