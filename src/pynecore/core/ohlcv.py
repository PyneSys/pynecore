"""OHLCV v2 storage with transparent read-only support for legacy v1 files."""

import csv
import json
import math
import mmap
import os
import re
import struct
import tempfile
from collections import Counter
from datetime import UTC, datetime, time, timedelta, timezone as fixed_timezone, tzinfo
from math import gcd as math_gcd
from pathlib import Path
from types import TracebackType
from typing import IO, BinaryIO, Iterator, NamedTuple, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pynecore.core.ohlcv_legacy import OHLCVReader as _LegacyOHLCVReader
from pynecore.core.syminfo import SymInfoInterval
from pynecore.types.ohlcv import OHLCV

__all__ = ["OHLCVWriter", "OHLCVReader", "parse_timezone_name", "record_count"]

_MAGIC = b"\x89PYN\r\n\x1a\n"
_VERSION_MAJOR = 2
_VERSION_MINOR = 0
_FIXED_HEADER_SIZE = 64
_DENSE_FLAG = 0x0001
_SUPPORTED_FLAGS = _DENSE_FLAG
_LEGACY_RECORD_SIZE = struct.calcsize("Ifffff")
_ABSOLUTE_BASE = 255
_CUSTOM_ROLE = 255

_TIMEZONE_OFFSET = re.compile(r"(UTC|GMT)?([+-])(\d{1,2}):?(\d{2})?", re.IGNORECASE)

_HEADER = struct.Struct("<8sHHIIHHQqqIB3x8x")
_DESCRIPTOR = struct.Struct("<BBBxH18s")
_DTYPE_FORMAT = {2: "q", 5: "f", 6: "d"}
_DTYPE_SIZE = {2: 8, 5: 4, 6: 8}
_SUPPORTED_ROLES = frozenset(range(14)) | {_CUSTOM_ROLE}
_FIXED_INTERVAL_MS = {
    1: 1_000,
    2: 60_000,
    3: 3_600_000,
    4: 86_400_000,
    5: 604_800_000,
}

_ROLE_TIMESTAMP = 0
_ROLE_OPEN = 1
_ROLE_HIGH = 2
_ROLE_LOW = 3
_ROLE_CLOSE = 4
_ROLE_VOLUME = 5
_REQUIRED_OHLCV_ROLES = frozenset(
    {_ROLE_TIMESTAMP, _ROLE_OPEN, _ROLE_HIGH, _ROLE_LOW, _ROLE_CLOSE, _ROLE_VOLUME}
)
_DELTA_CANDIDATE_ROLES = (_ROLE_HIGH, _ROLE_LOW, _ROLE_CLOSE)

_CANONICAL_NAN32 = struct.unpack("<f", bytes.fromhex("0000c07f"))[0]
_CANONICAL_NAN64 = struct.unpack("<d", bytes.fromhex("000000000000f87f"))[0]

_QTY_STEP_MIN_SAMPLES = 100
_QTY_STEP_MAX_DECIMALS = 8


class _Column(NamedTuple):
    role: int
    dtype: int
    base: int
    byte_offset: int
    name: str


class _Layout(NamedTuple):
    header_size: int
    record_size: int
    columns: tuple[_Column, ...]
    flags: int
    record_count: int
    first_timestamp: int
    last_timestamp: int
    interval_value: int
    interval_unit: int


class _WritableBinary(Protocol):
    def write(self, data: bytes, /) -> int | None:
        """Write bytes and return the number accepted."""


class _RowWriter(Protocol):
    def writerow(self, row: list[str], /) -> object:
        """Write one CSV row."""


def _extra_sidecar_path(path: str | Path) -> Path:
    """Return the position-aligned user extra-field sidecar path for an OHLCV file.

    :param path: OHLCV file path.
    :return: Matching ``.extra.csv`` sidecar path.
    """
    return Path(path).with_suffix(".extra.csv")


def _format_extra_float(value: float) -> str:
    """Format a sidecar float with the shared eight-significant-digit convention.

    :param value: Value to format.
    :return: Formatted text.
    """
    return f"{value:.8g}"


def _parse_period(period: str) -> tuple[str, int, int]:
    if not period:
        raise ValueError("OHLCV period cannot be empty")

    if period.isdigit():
        value = int(period)
        unit = 2
        canonical = str(value)
    else:
        suffix = period[-1]
        if suffix not in "SDWM":
            raise ValueError(f"Unsupported TradingView period: {period!r}")
        value_text = period[:-1]
        if not value_text or not value_text.isdigit():
            raise ValueError(f"Invalid TradingView period: {period!r}")
        value = int(value_text)
        unit = {"S": 1, "D": 4, "W": 5, "M": 6}[suffix]
        canonical = f"{value}{suffix}"

    if value <= 0:
        raise ValueError("OHLCV period multiplier must be positive")
    if value > 0xFFFFFFFF:
        raise ValueError("OHLCV period multiplier exceeds the v2 header range")
    return canonical, value, unit


def parse_timezone_name(name: str) -> tzinfo:
    """Parse an IANA timezone name or a numeric UTC offset.

    Both spellings travel through the data pipeline: importers accept ``+05:30``
    style offsets on the command line and symbol info stores whatever the source
    declared, so every consumer must understand the same set. Offsets keep their
    exact minutes instead of being rounded onto an ``Etc/GMT`` whole-hour zone.

    :param name: Timezone name such as ``UTC``, ``Europe/Budapest`` or ``UTC+05:30``.
    :return: Resolved timezone.
    :raises ValueError: If the name is empty or cannot be resolved.
    """
    if not name:
        raise ValueError("Timezone name cannot be empty")

    offset_match = _TIMEZONE_OFFSET.fullmatch(name.strip())
    if offset_match is not None:
        hours = int(offset_match.group(3))
        minutes = int(offset_match.group(4) or "0")
        if hours > 23 or minutes > 59:
            raise ValueError(f"Invalid UTC offset: {name!r}")
        sign = 1 if offset_match.group(2) == "+" else -1
        return fixed_timezone(sign * timedelta(hours=hours, minutes=minutes))

    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"Invalid timezone {name!r}: {error}") from error


def _resolve_timezone(name: str) -> tzinfo:
    """Resolve a timezone name, falling back to UTC when it is unusable.

    Opening-hours analysis is a best-effort convenience; an exotic or missing zone
    name must not make writing bars fail.

    :param name: Timezone name or numeric UTC offset.
    :return: Resolved timezone.
    """
    if not name or name.upper() in ("UTC", "GMT"):
        return UTC
    try:
        return parse_timezone_name(name)
    except ValueError:
        return UTC


def _period_from_interval(value: int, unit: int) -> str:
    if unit == 1:
        return f"{value}S"
    if unit == 2:
        return str(value)
    if unit == 3:
        return str(value * 60)
    if unit == 4:
        return f"{value}D"
    if unit == 5:
        return f"{value}W"
    if unit == 6:
        return f"{value}M"
    raise ValueError(f"Unsupported OHLCV interval unit: {unit}")


def _column_name(raw_name: bytes) -> str:
    nul_at = raw_name.find(b"\0")
    if nul_at >= 0:
        if any(raw_name[nul_at:]):
            raise ValueError("OHLCV column name has invalid NUL padding")
        raw_name = raw_name[:nul_at]
    if not raw_name:
        raise ValueError("OHLCV column name cannot be empty")
    try:
        return raw_name.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("OHLCV column name must be ASCII") from error


def _pack_descriptor(column: _Column) -> bytes:
    try:
        name = column.name.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"Invalid OHLCV column name: {column.name!r}") from error
    if not name or len(name) > 18 or b"\0" in name:
        raise ValueError(f"Invalid OHLCV column name: {column.name!r}")
    return _DESCRIPTOR.pack(
        column.role,
        column.dtype,
        column.base,
        column.byte_offset,
        name.ljust(18, b"\0"),
    )


def _make_default_columns(promoted_roles: frozenset[int] = frozenset()) -> tuple[_Column, ...]:
    definitions = (
        (_ROLE_TIMESTAMP, 2, _ABSOLUTE_BASE, "timestamp"),
        (_ROLE_OPEN, 6, _ABSOLUTE_BASE, "open"),
        (_ROLE_HIGH, 5, _ROLE_OPEN, "high"),
        (_ROLE_LOW, 5, _ROLE_OPEN, "low"),
        (_ROLE_CLOSE, 5, _ROLE_OPEN, "close"),
        (_ROLE_VOLUME, 6, _ABSOLUTE_BASE, "volume"),
    )
    columns: list[_Column] = []
    byte_offset = 0
    for role, dtype, base, name in definitions:
        if role in promoted_roles:
            dtype = 6
            base = _ABSOLUTE_BASE
        columns.append(_Column(role, dtype, base, byte_offset, name))
        byte_offset += _DTYPE_SIZE[dtype]
    return tuple(columns)


def _promote_columns(columns: tuple[_Column, ...], roles: frozenset[int]) -> tuple[_Column, ...]:
    promoted: list[_Column] = []
    byte_offset = 0
    for column in sorted(columns, key=lambda item: item.byte_offset):
        dtype = 6 if column.role in roles else column.dtype
        base = _ABSOLUTE_BASE if column.role in roles else column.base
        promoted.append(_Column(column.role, dtype, base, byte_offset, column.name))
        byte_offset += _DTYPE_SIZE[dtype]
    return tuple(promoted)


def _record_struct(columns: tuple[_Column, ...]) -> struct.Struct:
    ordered = sorted(columns, key=lambda item: item.byte_offset)
    return struct.Struct("<" + "".join(_DTYPE_FORMAT[item.dtype] for item in ordered))


def _build_header(
    columns: tuple[_Column, ...],
    flags: int,
    record_count: int,
    first_timestamp: int,
    last_timestamp: int,
    interval_value: int,
    interval_unit: int,
) -> bytes:
    header_size = _FIXED_HEADER_SIZE + _DESCRIPTOR.size * len(columns)
    record_size = sum(_DTYPE_SIZE[column.dtype] for column in columns)
    return _HEADER.pack(
        _MAGIC,
        _VERSION_MAJOR,
        _VERSION_MINOR,
        header_size,
        record_size,
        len(columns),
        flags,
        record_count,
        first_timestamp,
        last_timestamp,
        interval_value,
        interval_unit,
    )


def _validate_columns(columns: tuple[_Column, ...], record_size: int) -> None:
    seen_roles: set[int] = set()
    custom_names: set[str] = set()
    role_counts: dict[int, int] = {}

    for column in columns:
        if column.role not in _SUPPORTED_ROLES:
            raise ValueError(f"Unsupported OHLCV column role: {column.role}")
        if column.dtype not in _DTYPE_SIZE:
            raise ValueError(f"Unsupported OHLCV column dtype: {column.dtype}")
        if column.base != _ABSOLUTE_BASE and column.base not in range(14):
            raise ValueError(f"Unsupported OHLCV delta base: {column.base}")
        if column.role == _CUSTOM_ROLE:
            if column.name in custom_names:
                raise ValueError(f"Duplicate custom OHLCV column name: {column.name!r}")
            custom_names.add(column.name)
        elif column.role in seen_roles:
            raise ValueError(f"Duplicate OHLCV column role: {column.role}")
        else:
            seen_roles.add(column.role)
        role_counts[column.role] = role_counts.get(column.role, 0) + 1

    for column in columns:
        if column.base != _ABSOLUTE_BASE and role_counts.get(column.base) != 1:
            raise ValueError(
                f"OHLCV column {column.name!r} references a missing or ambiguous base role"
            )

    columns_by_role = {
        column.role: column for column in columns if column.role != _CUSTOM_ROLE
    }
    for column in columns_by_role.values():
        base = column.base
        seen = {column.role}
        while base != _ABSOLUTE_BASE:
            if base in seen:
                raise ValueError("OHLCV column delta bases contain a cycle")
            seen.add(base)
            base = columns_by_role[base].base

    timestamp_columns = [column for column in columns if column.role == _ROLE_TIMESTAMP]
    if len(timestamp_columns) != 1:
        raise ValueError("OHLCV schema must contain exactly one timestamp column")
    timestamp = timestamp_columns[0]
    if timestamp.dtype != 2 or timestamp.base != _ABSOLUTE_BASE:
        raise ValueError("OHLCV timestamp must be an absolute i64 value")

    ordered = sorted(columns, key=lambda item: item.byte_offset)
    expected_offset = 0
    for column in ordered:
        if column.byte_offset != expected_offset:
            raise ValueError("OHLCV column ranges must tile each packed record without gaps")
        expected_offset += _DTYPE_SIZE[column.dtype]
    if expected_offset != record_size:
        raise ValueError("OHLCV column ranges do not match record_size")


def _read_layout(file: BinaryIO, file_size: int, magic: bytes | None = None) -> _Layout:
    if file_size < _FIXED_HEADER_SIZE:
        raise ValueError("File is too short to contain an OHLCV v2 header")
    if magic is None:
        raw_header = os.pread(file.fileno(), _FIXED_HEADER_SIZE, 0)
    else:
        raw_header = magic + os.pread(
            file.fileno(), _FIXED_HEADER_SIZE - len(_MAGIC), len(_MAGIC)
        )
    if len(raw_header) != _FIXED_HEADER_SIZE:
        raise ValueError("Incomplete OHLCV v2 header")

    (
        magic,
        version_major,
        version_minor,
        header_size,
        record_size,
        column_count,
        flags,
        record_count,
        first_timestamp,
        last_timestamp,
        interval_value,
        interval_unit,
    ) = _HEADER.unpack(raw_header)

    if magic != _MAGIC:
        raise ValueError("Invalid OHLCV v2 magic; the file is not in v2 format")
    if version_major != _VERSION_MAJOR:
        raise ValueError(f"Unsupported OHLCV major version: {version_major}")
    if version_minor != _VERSION_MINOR:
        raise ValueError(f"Unsupported OHLCV minor version: {version_minor}")
    if flags & ~_SUPPORTED_FLAGS:
        raise ValueError(f"Unsupported OHLCV header flags: 0x{flags:04x}")
    if column_count == 0:
        raise ValueError("OHLCV schema cannot be empty")
    expected_header_size = _FIXED_HEADER_SIZE + column_count * _DESCRIPTOR.size
    if header_size != expected_header_size:
        raise ValueError("OHLCV header_size does not match the descriptor count")
    if record_size == 0:
        raise ValueError("OHLCV record_size must be positive")
    if interval_value <= 0:
        raise ValueError("OHLCV interval_value must be positive")
    if interval_unit not in range(1, 7):
        raise ValueError(f"Unsupported OHLCV interval unit: {interval_unit}")
    if file_size < header_size:
        raise ValueError("OHLCV file is shorter than its declared header")

    raw_descriptors = os.pread(file.fileno(), column_count * _DESCRIPTOR.size, _FIXED_HEADER_SIZE)
    if len(raw_descriptors) != column_count * _DESCRIPTOR.size:
        raise ValueError("Incomplete OHLCV descriptor block")
    columns: list[_Column] = []
    for index in range(column_count):
        start = index * _DESCRIPTOR.size
        role, dtype, base, byte_offset, raw_name = _DESCRIPTOR.unpack_from(raw_descriptors, start)
        columns.append(_Column(role, dtype, base, byte_offset, _column_name(raw_name)))
    column_tuple = tuple(columns)
    _validate_columns(column_tuple, record_size)

    committed_end = header_size + record_count * record_size
    if committed_end < header_size or committed_end > file_size:
        raise ValueError("OHLCV file is missing data committed by record_count")

    timestamp_column = next(column for column in column_tuple if column.role == _ROLE_TIMESTAMP)
    timestamp_struct = struct.Struct("<q")
    if record_count == 0:
        if first_timestamp != 0 or last_timestamp != 0:
            raise ValueError("Empty OHLCV files must have zero first and last timestamps")
        if flags & _DENSE_FLAG:
            raise ValueError("Empty OHLCV files cannot be marked DENSE")
    else:
        first_offset = header_size + timestamp_column.byte_offset
        last_offset = header_size + (record_count - 1) * record_size + timestamp_column.byte_offset
        actual_first = timestamp_struct.unpack(os.pread(file.fileno(), 8, first_offset))[0]
        actual_last = timestamp_struct.unpack(os.pread(file.fileno(), 8, last_offset))[0]
        if first_timestamp != actual_first or last_timestamp != actual_last:
            raise ValueError("OHLCV header timestamps do not match the committed records")
        if record_count == 1 and flags & _DENSE_FLAG:
            raise ValueError("A one-record OHLCV file cannot be marked DENSE")

    return _Layout(
        header_size,
        record_size,
        column_tuple,
        flags,
        record_count,
        first_timestamp,
        last_timestamp,
        interval_value,
        interval_unit,
    )


def _validate_candle(candle: OHLCV) -> None:
    values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
    if any(math.isinf(value) for value in values):
        raise ValueError("OHLCV values cannot contain positive or negative infinity")

    if not (math.isnan(candle.open) or math.isnan(candle.close) or math.isnan(candle.high)):
        if candle.high < max(candle.open, candle.close):
            raise ValueError("Invalid OHLC relation: high is below open or close")
    if not (math.isnan(candle.open) or math.isnan(candle.close) or math.isnan(candle.low)):
        if candle.low > min(candle.open, candle.close):
            raise ValueError("Invalid OHLC relation: low is above open or close")


def _f32_resolution(value: float) -> float | None:
    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        return None
    rounded = struct.unpack("<f", packed)[0]
    if not math.isfinite(rounded):
        return None
    bits = struct.unpack("<I", packed)[0]

    if bits in (0, 0x80000000):
        lower_bits = 0x80000001
        upper_bits = 0x00000001
    elif bits & 0x80000000:
        if bits >= 0xFF7FFFFF:
            return None
        lower_bits = bits + 1
        upper_bits = bits - 1
    else:
        if bits >= 0x7F7FFFFF:
            return None
        lower_bits = bits - 1
        upper_bits = bits + 1

    lower = struct.unpack("<f", struct.pack("<I", lower_bits))[0]
    upper = struct.unpack("<f", struct.pack("<I", upper_bits))[0]
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return max(rounded - lower, upper - rounded)


def _failing_delta_roles(candle: OHLCV, mintick: float | None) -> frozenset[int]:
    """
    Select the price columns whose f32 delta cannot resolve half a tick.

    Without a tick size there is nothing to measure against, and f32 deltas are the
    correct default: no tradable instrument in the measured corpus exceeds half a
    mintick with delta encoding, while promoting on absence would inflate the base
    profile from 36 to 48 bytes per record.

    :param candle: Bar the decision is measured on.
    :param mintick: Tick size to verify against, or ``None`` to keep f32 deltas.
    :return: Roles that must be stored as absolute f64.
    """
    if mintick is None:
        return frozenset()

    failed: set[int] = set()
    half_tick = mintick / 2.0
    for role, target in (
        (_ROLE_HIGH, candle.high),
        (_ROLE_LOW, candle.low),
        (_ROLE_CLOSE, candle.close),
    ):
        if math.isnan(target):
            continue
        if math.isnan(candle.open):
            failed.add(role)
            continue
        resolution = _f32_resolution(target - candle.open)
        if resolution is None or resolution >= half_tick:
            failed.add(role)
    return frozenset(failed)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_all(file: _WritableBinary, data: bytes, message: str) -> None:
    written = 0
    while written < len(data):
        count = file.write(data[written:])
        if count is None or count <= 0:
            raise OSError(message)
        written += count


def _pwrite_all(fd: int, data: bytes, offset: int, message: str) -> None:
    written = 0
    while written < len(data):
        count = os.pwrite(fd, data[written:], offset + written)
        if count <= 0:
            raise OSError(message)
        written += count


def record_count(path: str | Path) -> int:
    """Return how many records an OHLCV file holds, whatever its on-disk format.

    The eight-byte magic is the sole format discriminator, exactly as in
    :meth:`OHLCVReader.open`. A v2 file reports the authoritative ``record_count``
    from its fixed header, so bytes past the committed end are never counted. A
    legacy v1 file reports its size divided by the v1 record size.

    This is a cheap pre-flight probe meant for cache gates: it reads a few hundred
    header bytes, opens no reader, maps nothing, and raises nothing. Zero means "no
    records to use here" and covers a missing path, an empty file, a file too short
    to hold a header, an unreadable file, a v2 file whose header or schema fails
    validation, and any non-v2 file whose size is not a whole number of v1 records
    — for example a CSV saved under an ``.ohlcv`` name.

    :param path: OHLCV file path to probe.
    :return: Number of records, or zero when the file holds no usable records.
    """
    try:
        file = open(path, "rb", buffering=0)
    except OSError:
        return 0

    try:
        file_size = os.fstat(file.fileno()).st_size
        magic = file.read(len(_MAGIC))
        if magic == _MAGIC:
            try:
                return _read_layout(file, file_size, magic).record_count
            except (OSError, ValueError, struct.error):
                return 0
        if file_size == 0 or file_size % _LEGACY_RECORD_SIZE != 0:
            return 0
        return file_size // _LEGACY_RECORD_SIZE
    except OSError:
        return 0
    finally:
        file.close()


class OHLCVWriter:
    """Write committed OHLCV v2 records with durable append publication.

    :param path: Target OHLCV file path.
    :param period: Declared nominal timeframe in TradingView format.
    :param mintick: Positive tick size used to verify f32 delta resolution.
    :param truncate: Replace any existing file with a new empty v2 file.
    """

    __slots__ = (
        "path",
        "_period",
        "_interval_value",
        "_interval_unit",
        "_truncate_requested",
        "_file",
        "_columns",
        "_record_struct",
        "_ordered_columns",
        "_header_size",
        "_record_size",
        "_size",
        "_start_timestamp",
        "_last_timestamp",
        "_dense",
        "_mintick",
        "_price_changes",
        "_price_decimals",
        "_last_analyzed_close",
        "_analyzed_tick_size",
        "_analyzed_price_scale",
        "_analyzed_min_move",
        "_confidence",
        "_volume_max_decimals",
        "_volume_count",
        "_volume_dust_count",
        "_trading_hours",
        "_trading_hours_tz",
        "_analyzed_opening_hours",
        "_extra_path",
        "_extra_file",
        "_extra_writer",
        "_extra_headers",
        "_extra_row_count",
        "_rebuild_published",
    )

    def __init__(
        self,
        path: str | Path,
        period: str,
        *,
        mintick: float | None = None,
        truncate: bool = False,
        timezone: str = "UTC",
    ):
        """Create a closed OHLCV v2 writer.

        Timestamps are Unix milliseconds, both in the accepted
        :class:`~pynecore.types.ohlcv.OHLCV` values and in the stored ``timestamp``
        field.

        :param path: Target OHLCV file path.
        :param period: Declared nominal timeframe in TradingView format.
        :param mintick: Optional positive tick size for initial f32-delta fallback selection.
        :param truncate: Replace any existing file with a new empty v2 file.
        :param timezone: IANA name of the symbol timezone the opening-hours analysis
            reports its weekday and hour intervals in. Unknown names fall back to UTC.
        :raises ValueError: If ``period`` or ``mintick`` is invalid.
        """
        canonical, interval_value, interval_unit = _parse_period(period)
        if mintick is not None and (not math.isfinite(mintick) or mintick <= 0.0):
            raise ValueError("OHLCV mintick must be finite and positive")
        self.path = str(path)
        self._period = canonical
        self._interval_value = interval_value
        self._interval_unit = interval_unit
        self._truncate_requested = truncate
        self._file: BinaryIO | None = None
        self._columns = _make_default_columns()
        self._record_struct = _record_struct(self._columns)
        self._ordered_columns = tuple(sorted(self._columns, key=lambda column: column.byte_offset))
        self._header_size = _FIXED_HEADER_SIZE + len(self._columns) * _DESCRIPTOR.size
        self._record_size = self._record_struct.size
        self._size = 0
        self._start_timestamp: int | None = None
        self._last_timestamp: int | None = None
        self._dense = False
        self._mintick = mintick
        self._price_changes: list[float] = []
        self._price_decimals: set[int] = set()
        self._last_analyzed_close: float | None = None
        self._analyzed_tick_size: float | None = None
        self._analyzed_price_scale: int | None = None
        self._analyzed_min_move: int | None = None
        self._confidence = 0.0
        self._volume_max_decimals = 0
        self._volume_count = 0
        self._volume_dust_count = 0
        self._trading_hours: dict[tuple[int, int], int] = {}
        self._trading_hours_tz: tzinfo = _resolve_timezone(timezone)
        self._analyzed_opening_hours: list[SymInfoInterval] | None = None
        self._extra_path = _extra_sidecar_path(path)
        self._extra_file: IO[str] | None = None
        self._extra_writer: _RowWriter | None = None
        self._extra_headers: list[str] | None = None
        self._extra_row_count = 0
        self._rebuild_published = False

    def __enter__(self) -> "OHLCVWriter":
        """Open the writer and return it.

        :return: This writer.
        """
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the writer on every context-manager exit path.

        :param exc_type: Exception type raised in the context, if any.
        :param exc_val: Exception value raised in the context, if any.
        :param exc_tb: Exception traceback raised in the context, if any.
        """
        self.close()

    @property
    def size(self) -> int:
        """Return the authoritative number of committed records.

        :return: Committed record count.
        """
        return self._size

    @property
    def period(self) -> str:
        """Return the declared canonical TradingView timeframe.

        :return: Canonical file period.
        """
        return self._period

    @property
    def start_timestamp(self) -> int | None:
        """Return the first committed timestamp in milliseconds.

        :return: First timestamp, or ``None`` for an empty file.
        """
        return self._start_timestamp

    @property
    def start_datetime(self) -> datetime:
        """Return the first committed timestamp as a UTC datetime.

        :return: First timestamp converted from milliseconds.
        :raises AssertionError: If the file is empty.
        """
        assert self._start_timestamp is not None
        return datetime.fromtimestamp(self._start_timestamp / 1000, UTC)

    @property
    def end_timestamp(self) -> int | None:
        """Return the actual last committed timestamp in milliseconds.

        :return: Last timestamp, or ``None`` for an empty file.
        """
        return self._last_timestamp

    @property
    def analyzed_tick_size(self) -> float | None:
        """Return the automatically detected tick size.

        :return: Estimated tick size, or ``None`` without enough price changes.
        """
        if self._analyzed_tick_size is None and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._analyzed_tick_size

    @property
    def analyzed_price_scale(self) -> int | None:
        """Return the automatically detected price scale.

        :return: Estimated price scale, or ``None`` without enough price changes.
        """
        if self._analyzed_price_scale is None and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._analyzed_price_scale

    @property
    def analyzed_min_move(self) -> int | None:
        """Return the automatically detected minimum price move.

        :return: Estimated minimum move, or ``None`` without enough price changes.
        """
        if self._analyzed_min_move is None and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._analyzed_min_move

    @property
    def tick_analysis_confidence(self) -> float:
        """Return the tick-size analysis confidence from zero to one.

        :return: Tick-size analysis confidence.
        """
        if self._confidence == 0.0 and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._confidence

    @property
    def analyzed_opening_hours(self) -> list[SymInfoInterval] | None:
        """Return opening hours inferred from written bar timestamps.

        :return: Opening-hour intervals, or ``None`` without enough data.
        """
        if self._analyzed_opening_hours is None and self._has_enough_data_for_opening_hours():
            self._analyze_opening_hours()
        return self._analyzed_opening_hours

    @property
    def analyzed_qty_step(self) -> float | None:
        """Return a ``mincontract`` candidate inferred from positive volumes.

        The analysis uses original float64 values rather than encoded records. NaN and
        non-positive volumes carry no quantity-grid information and are excluded.

        :return: Estimated quantity step, or ``None`` without reliable samples.
        """
        if self._volume_count < _QTY_STEP_MIN_SAMPLES:
            return None
        if self._volume_dust_count * 20 > self._volume_count:
            return None
        return 10.0 ** -self._volume_max_decimals

    def open(self) -> "OHLCVWriter":
        """Open, create, or validate the target v2 file for append.

        Uncommitted bytes beyond the count-derived committed end are truncated before
        append. The method never dispatches to the v1 format.

        :return: This writer.
        :raises ValueError: If an existing file has an incompatible header, period, or schema.
        """
        if self._file is not None:
            return self

        path = Path(self.path)
        path_exists = path.exists()
        existing_magic: bytes | None = None
        if path_exists:
            with open(path, "rb", buffering=0) as existing_file:
                existing_magic = existing_file.read(len(_MAGIC))
            if existing_magic != _MAGIC and not self._truncate_requested:
                raise ValueError(
                    "Cannot write to a legacy v1 OHLCV file; re-download the data to replace it "
                    "with v2"
                )

        if self._truncate_requested or not path_exists:
            file = open(path, "w+b", buffering=0)
            self._file = file
            try:
                self._set_columns(_make_default_columns())
                self._size = 0
                self._start_timestamp = None
                self._last_timestamp = None
                self._dense = False
                self._write_empty_file()
                self._discard_extra_csv()
            except Exception:
                file.close()
                self._file = None
                self._truncate_requested = True
                raise
            self._truncate_requested = False
            return self

        file = open(path, "r+b", buffering=0)
        self._file = file
        try:
            file_size = os.fstat(file.fileno()).st_size
            assert existing_magic is not None
            layout = _read_layout(file, file_size, existing_magic)
            file_period = _period_from_interval(layout.interval_value, layout.interval_unit)
            if file_period != self._period:
                raise ValueError(
                    f"OHLCV period mismatch: file declares {file_period!r}, writer uses {self._period!r}"
                )
            self._validate_writer_schema(layout.columns)
            committed_end = layout.header_size + layout.record_count * layout.record_size
            if file_size > committed_end:
                file.truncate(committed_end)
                file.flush()
                os.fsync(file.fileno())
            self._set_columns(layout.columns)
            self._header_size = layout.header_size
            self._record_size = layout.record_size
            self._size = layout.record_count
            self._start_timestamp = layout.first_timestamp if layout.record_count else None
            self._last_timestamp = layout.last_timestamp if layout.record_count else None
            self._dense = bool(layout.flags & _DENSE_FLAG)
            self._interval_value = layout.interval_value
            self._interval_unit = layout.interval_unit
            # The sidecar is adopted (and realigned when an interrupted append left
            # it short) before anything reads the file back: the strict reader used
            # for the trading-hours scan rejects a misaligned pair outright, so the
            # repair would never be reached the other way round.
            self._adopt_extra_csv()
            if self._size > 0:
                self._collect_existing_trading_hours()
        except Exception:
            file.close()
            self._file = None
            raise
        return self

    def close(self) -> None:
        """Flush and close the file; repeated calls are safe."""
        self._close_extra_csv()
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None

    def write(self, candle: OHLCV) -> None:
        """Validate and append one real OHLCV record.

        The supplied ``OHLCV.timestamp`` is a signed i64 Unix millisecond value. No gap
        records are inserted. Record bytes are flushed and fsynced before the header's
        authoritative ``record_count`` is increased.

        Any ``extra_fields`` are appended to the position-aligned ``.extra.csv``
        sidecar before the record is published, so sidecar data row ``N`` always
        describes committed record ``N`` and a failed write commits nothing.

        :param candle: Logical OHLCV bar with a millisecond timestamp.
        :raises RuntimeError: If the writer is closed.
        :raises ValueError: If the bar is malformed or its timestamp is not strictly increasing.
        """
        if self._file is None:
            raise RuntimeError("OHLCV writer is not open")
        _validate_candle(candle)
        if not -(1 << 63) <= candle.timestamp < (1 << 63):
            raise ValueError("OHLCV timestamp is outside the signed i64 range")
        if self._last_timestamp is not None and candle.timestamp <= self._last_timestamp:
            if candle.timestamp == self._last_timestamp:
                raise ValueError(f"Duplicate OHLCV timestamp: {candle.timestamp}")
            raise ValueError(
                "OHLCV timestamps must be strictly increasing: "
                f"{candle.timestamp} follows {self._last_timestamp}"
            )

        f32_roles = frozenset(
            column.role
            for column in self._columns
            if column.dtype == 5 and column.base == _ROLE_OPEN
        )
        old_count = self._size
        promoted_roles = _failing_delta_roles(candle, self._mintick) & f32_roles
        if promoted_roles:
            if self._size == 0:
                self._set_columns(_promote_columns(self._columns, promoted_roles))
                self._write_empty_file()
            else:
                self._commit_extra_row(candle.extra_fields, old_count)
                self._rebuild_published = False
                try:
                    self._rebuild_with_append(candle, promoted_roles)
                except Exception:
                    # Once the rebuilt file is in place the bar is committed, so its
                    # sidecar row has to stay: trimming it would leave the pair one row
                    # short, which is unreadable and pads the lost values away on the
                    # next append. Only a failure before publication is undone here.
                    if not self._rebuild_published:
                        self._rollback_extra_row(old_count)
                    raise
                return

        record = self._pack_record(candle)
        record_offset = self._header_size + old_count * self._record_size
        if os.pwrite(self._file.fileno(), record, record_offset) != len(record):
            raise OSError("Could not write the complete OHLCV record")
        os.fsync(self._file.fileno())

        # The sidecar row is durable before the record is published: the header still
        # names ``old_count`` records, so a failing sidecar write leaves the bar
        # uncommitted and the caller can retry it with its extra fields intact. The
        # other way round the bar would be committed with its extra values lost, and
        # the retry rejected as a duplicate timestamp.
        self._commit_extra_row(candle.extra_fields, old_count)

        first_timestamp = self._start_timestamp if self._start_timestamp is not None else candle.timestamp
        new_count = old_count + 1
        dense = self._dense_after_append(candle.timestamp)
        flags = _DENSE_FLAG if dense else 0
        previous_header = _build_header(
            self._columns,
            _DENSE_FLAG if self._dense else 0,
            old_count,
            self._start_timestamp if self._start_timestamp is not None else 0,
            self._last_timestamp if self._last_timestamp is not None else 0,
            self._interval_value,
            self._interval_unit,
        )
        header = _build_header(
            self._columns,
            flags,
            new_count,
            first_timestamp,
            candle.timestamp,
            self._interval_value,
            self._interval_unit,
        )
        if os.pwrite(self._file.fileno(), header, 0) != len(header):
            _pwrite_all(
                self._file.fileno(),
                previous_header,
                0,
                "Could not restore the previous OHLCV header",
            )
            os.fsync(self._file.fileno())
            self._rollback_extra_row(old_count)
            raise OSError("Could not publish the complete OHLCV header")
        os.fsync(self._file.fileno())

        self._size = new_count
        self._start_timestamp = first_timestamp
        self._last_timestamp = candle.timestamp
        self._dense = dense
        self._collect_price_data(candle)
        self._collect_volume_data(candle)
        self._collect_trading_hours(candle)

    def truncate(self) -> None:
        """Reset the open file to an empty valid v2 file with the same schema and period.

        The position-aligned ``.extra.csv`` sidecar is deleted with the records it
        described; leaving it behind would make the pair unreadable, because the reader
        requires exactly one sidecar data row per committed record.

        :raises RuntimeError: If the writer is closed.
        """
        if self._file is None:
            raise RuntimeError("OHLCV writer is not open")
        self._size = 0
        self._start_timestamp = None
        self._last_timestamp = None
        self._dense = False
        self._price_changes.clear()
        self._price_decimals.clear()
        self._last_analyzed_close = None
        self._analyzed_tick_size = None
        self._analyzed_price_scale = None
        self._analyzed_min_move = None
        self._confidence = 0.0
        self._volume_max_decimals = 0
        self._volume_count = 0
        self._volume_dust_count = 0
        self._trading_hours.clear()
        self._analyzed_opening_hours = None
        self._write_empty_file()
        self._discard_extra_csv()

    def _adopt_extra_csv(self) -> None:
        """Adopt an existing sidecar so later appends extend it in place.

        Only the header and the data-row count are needed: appended rows are placed
        after the rows already present, and any shortfall is padded when the first
        bar carrying extra fields arrives.
        """
        self._extra_headers = None
        self._extra_row_count = 0
        if not self._extra_path.exists():
            return
        with open(self._extra_path, "r", encoding="utf-8-sig", newline="") as extra_file:
            reader = csv.reader(extra_file)
            headers = next(reader, None)
            if not headers:
                return
            self._extra_headers = headers
            self._extra_row_count = sum(1 for _ in reader)
        if self._extra_row_count != self._size:
            # An interrupted append left the pair misaligned, and the reader accepts
            # nothing but one data row per record — repair it here, otherwise the
            # whole dataset stays unreadable until the sidecar is deleted by hand.
            self._align_extra_csv(self._size)

    def _align_extra_csv(self, records: int) -> None:
        """Rewrite the sidecar so it holds exactly ``records`` well-formed data rows.

        Surplus rows are dropped, missing ones are appended empty, and a partially
        written row ends the usable prefix (everything after it is replaced by empty
        rows). Position alignment is preserved for every row that survives, so bars
        keep the extra values they were written with.

        :param records: Number of committed records the sidecar must describe.
        """
        if self._extra_headers is None:
            return
        self._close_extra_csv()
        headers = self._extra_headers
        kept: list[list[str]] = []
        if self._extra_path.exists():
            with open(self._extra_path, "r", encoding="utf-8-sig", newline="") as extra_file:
                reader = csv.reader(extra_file)
                next(reader, None)
                for row in reader:
                    if len(kept) >= records or len(row) != len(headers):
                        break
                    kept.append(row)
        empty = [""] * len(headers)
        with open(self._extra_path, "w", encoding="utf-8", newline="") as extra_file:
            writer = csv.writer(extra_file)
            writer.writerow(headers)
            for row in kept:
                writer.writerow(row)
            for _ in range(records - len(kept)):
                writer.writerow(empty)
        self._extra_row_count = records

    def _commit_extra_row(self, extra_fields: dict[str, object] | None, position: int) -> None:
        """Append the sidecar row of the record that is about to be published.

        The row is written while the binary still holds ``position`` records, so a
        failure can be undone completely: the sidecar is trimmed back to the
        committed record count and the exception propagates with nothing written.

        :param extra_fields: The bar's extra fields, if any.
        :param position: Zero-based index of the record the row describes, which is
            also the number of records committed so far.
        """
        try:
            self._append_extra_row(extra_fields, position)
        except Exception:
            self._rollback_extra_row(position)
            raise

    def _rollback_extra_row(self, records: int) -> None:
        """Trim the sidecar back to ``records`` data rows after a failed append.

        The reader accepts nothing but one data row per committed record, so a
        surplus or missing row would make the whole pair unreadable. This runs while
        another failure is being propagated, so an unwritable sidecar is swallowed:
        the original error is the one worth reporting.

        :param records: Number of committed records the sidecar must describe.
        """
        try:
            self._align_extra_csv(records)
        except OSError:
            pass

    def _discard_extra_csv(self) -> None:
        """Delete the sidecar and forget its shape after the records it described."""
        self._close_extra_csv()
        self._extra_path.unlink(missing_ok=True)
        self._extra_headers = None
        self._extra_row_count = 0

    def _close_extra_csv(self) -> None:
        """Close the sidecar file if it is open; repeated calls are safe."""
        if self._extra_file is not None:
            self._extra_file.close()
            self._extra_file = None
            self._extra_writer = None

    def _open_extra_csv(self, headers: list[str], position: int) -> None:
        """Open the sidecar for append, padding it out to ``position`` data rows.

        An existing sidecar with the same header is extended; any other header cannot
        describe the same columns, so the sidecar is rewritten from scratch. Records
        already committed without extra fields are represented by empty rows, which
        keeps data row ``N`` aligned with record ``N``.

        :param headers: Extra-field names in the order the first carrying bar declared.
        :param position: Zero-based index of the record the next row will describe.
        """
        empty = [""] * len(headers)
        if self._extra_headers == headers and self._extra_row_count <= position:
            extra_file = open(self._extra_path, "a", encoding="utf-8", newline="")
            writer = csv.writer(extra_file)
        else:
            extra_file = open(self._extra_path, "w", encoding="utf-8", newline="")
            writer = csv.writer(extra_file)
            writer.writerow(headers)
            self._extra_headers = headers
            self._extra_row_count = 0
        for _ in range(position - self._extra_row_count):
            writer.writerow(empty)
        self._extra_row_count = position
        self._extra_file = extra_file
        self._extra_writer = writer

    def _append_extra_row(self, extra_fields: dict[str, object] | None, position: int) -> None:
        """Append the sidecar row describing the record just committed at ``position``.

        Nothing is written until the first bar carrying extra fields arrives; from that
        point every record gets a row, empty when the bar carries no extra fields, so
        the sidecar keeps exactly one data row per committed record. A sidecar adopted
        from a reopened file counts as "already arrived": its rows must keep growing
        with the records even while the appended bars carry no extra fields, otherwise
        the reader rejects the pair for a row-count mismatch.

        :param extra_fields: The committed bar's extra fields, if any.
        :param position: Zero-based index of the record the row describes.
        """
        if self._extra_writer is None:
            if extra_fields:
                self._open_extra_csv(list(extra_fields.keys()), position)
            elif self._extra_headers is not None:
                self._open_extra_csv(list(self._extra_headers), position)
            else:
                return
        assert self._extra_writer is not None
        assert self._extra_headers is not None
        row: list[str] = []
        for header in self._extra_headers:
            value = None if not extra_fields else extra_fields.get(header)
            if value is None:
                row.append("")
            elif isinstance(value, float):
                row.append(_format_extra_float(value))
            else:
                row.append(str(value))
        self._extra_writer.writerow(row)
        self._extra_row_count += 1
        assert self._extra_file is not None
        self._extra_file.flush()

    def _collect_price_data(self, candle: OHLCV) -> None:
        """Collect bounded price statistics for tick-size analysis.

        :param candle: Newly committed real bar.
        """
        close = candle.close
        if self._last_analyzed_close is not None and math.isfinite(close):
            change = abs(close - self._last_analyzed_close)
            if change > 0.0 and len(self._price_changes) < 1000:
                self._price_changes.append(change)

        for price in (candle.open, candle.high, candle.low, candle.close):
            if not math.isfinite(price) or price == int(price):
                continue
            price_text = f"{price:.15f}".rstrip("0").rstrip(".")
            if "." in price_text:
                self._price_decimals.add(len(price_text.split(".", 1)[1]))

        self._last_analyzed_close = close if math.isfinite(close) else None

    def _collect_volume_data(self, candle: OHLCV) -> None:
        """Collect decimal precision from original positive volume values.

        :param candle: Newly committed real bar.
        """
        volume = candle.volume
        if math.isnan(volume) or volume <= 0.0:
            return
        volume_text = str(volume)
        if "e" in volume_text or "E" in volume_text:
            volume_text = f"{volume:.20f}".rstrip("0")
        decimals = (
            len(volume_text.split(".", 1)[1].rstrip("0")) if "." in volume_text else 0
        )
        if decimals > _QTY_STEP_MAX_DECIMALS:
            self._volume_dust_count += 1
            return
        self._volume_count += 1
        if decimals > self._volume_max_decimals:
            self._volume_max_decimals = decimals

    def _analyze_tick_size(self) -> None:
        """Analyze collected price changes with complementary estimators."""
        if not self._price_changes:
            self._analyzed_tick_size = 0.01
            self._analyzed_price_scale = 100
            self._analyzed_min_move = 1
            self._confidence = 0.1
            return

        histogram_tick = self._calculate_histogram_tick()
        if histogram_tick[0] > 0.0 and histogram_tick[1] > 0.7:
            self._analyzed_tick_size = histogram_tick[0]
            self._analyzed_price_scale = int(round(1.0 / histogram_tick[0]))
            self._analyzed_min_move = 1
            self._confidence = histogram_tick[1]
            return

        frequency_tick = self._calculate_frequency_tick()
        decimal_tick = self._calculate_decimal_tick()
        tick_size, confidence = self._combine_tick_estimates(frequency_tick, decimal_tick)
        if tick_size > 0.0:
            self._analyzed_tick_size = tick_size
            self._analyzed_price_scale = int(round(1.0 / tick_size))
            self._analyzed_min_move = 1
            self._confidence = confidence
            return

        self._analyzed_tick_size = 0.01
        self._analyzed_price_scale = 100
        self._analyzed_min_move = 1
        self._confidence = 0.1

    def _calculate_frequency_tick(self) -> tuple[float, float]:
        """Estimate tick size from repeated small price changes.

        :return: ``(tick_size, confidence)``.
        """
        if len(self._price_changes) < 10:
            return 0.0, 0.0

        filtered_changes: list[float] = []
        for change in self._price_changes[:100]:
            if change <= 0.0:
                continue
            float32_value = struct.unpack("<f", struct.pack("<f", change))[0]
            rounded = round(float32_value, 6)
            if rounded > 0.0:
                filtered_changes.append(rounded)

        if len(filtered_changes) < 5:
            return 0.0, 0.0

        most_common = Counter(filtered_changes).most_common(10)
        frequent_changes = [change for change, count in most_common if count >= 2]
        if len(frequent_changes) < 2:
            return 0.0, 0.0

        scale = 1_000_000
        integer_changes = [int(round(change * scale)) for change in frequent_changes]
        result = integer_changes[0]
        for value in integer_changes[1:]:
            result = math_gcd(result, value)
        tick_size = result / scale
        matches = sum(
            1
            for change in filtered_changes
            if abs(round(change / tick_size) * tick_size - change) < tick_size * 0.1
        )
        confidence = min(matches / len(filtered_changes), 1.0)
        return tick_size, confidence * 0.7

    def _calculate_histogram_tick(self) -> tuple[float, float]:
        """Estimate tick size by fitting common grids to price changes.

        :return: ``(tick_size, confidence)``.
        """
        if len(self._price_changes) < 10:
            return 0.0, 0.0

        candidate_ticks = (
            1.0,
            0.5,
            0.25,
            0.1,
            0.05,
            0.01,
            0.005,
            0.001,
            0.0005,
            0.0001,
            0.00005,
            0.00001,
            0.000001,
        )
        changes = [
            struct.unpack("<f", struct.pack("<f", change))[0]
            for change in self._price_changes[:200]
            if change > 0.0
        ]
        if len(changes) < 5:
            return 0.0, 0.0

        minimum_change = min(changes)
        average_change = sum(changes) / len(changes)
        best_tick = 0.0
        best_score = 0.0

        for tick in candidate_ticks:
            if tick < minimum_change * 0.1 or tick > average_change * 10.0:
                continue
            rounded = [round(change / tick) * tick for change in changes]
            maximum_error = max(
                abs(change - rounded_change)
                for change, rounded_change in zip(changes, rounded)
            )
            if maximum_error >= tick * 0.5:
                continue
            tolerance = tick * 0.1
            multiples = sum(
                1
                for change in changes
                if abs(round(change / tick) * tick - change) < tolerance
            )
            multiple_ratio = multiples / len(changes)
            if multiple_ratio <= 0.7:
                continue
            score = multiple_ratio * (1.0 + tick * 100.0)
            if score > best_score:
                best_score = score
                best_tick = tick

        if best_tick == 0.0:
            magnitudes = [10 ** math.floor(math.log10(change)) for change in changes]
            if magnitudes:
                best_tick = Counter(magnitudes).most_common(1)[0][0] / 10.0
                best_score = 0.5

        if best_score > 0.8:
            confidence = 0.9
        elif best_score > 0.6:
            confidence = 0.7
        else:
            confidence = best_score
        return best_tick, confidence

    def _calculate_decimal_tick(self) -> tuple[float, float]:
        """Estimate tick size from observed price decimal places.

        :return: ``(tick_size, confidence)``.
        """
        if not self._price_decimals:
            return 1.0, 0.5
        valid_decimals = [decimals for decimals in self._price_decimals if decimals <= 10]
        if not valid_decimals:
            return 0.01, 0.3
        return 10 ** -max(valid_decimals), 0.5

    @staticmethod
    def _combine_tick_estimates(
        frequency: tuple[float, float], decimal: tuple[float, float]
    ) -> tuple[float, float]:
        """Select the highest-confidence tick estimate.

        :param frequency: Frequency-based ``(tick_size, confidence)``.
        :param decimal: Decimal-based ``(tick_size, confidence)``.
        :return: Selected ``(tick_size, confidence)``.
        """
        estimates = [
            estimate
            for estimate in (frequency, decimal)
            if estimate[0] > 0.0 and estimate[1] > 0.0
        ]
        if not estimates:
            return 0.01, 0.1
        return max(estimates, key=lambda estimate: estimate[1])

    def _collect_trading_hours(self, candle: OHLCV) -> None:
        """Collect weekday and hour activity from a real v2 bar.

        V2 stores no phantom gap records, so volume is not used as a reality marker.
        Activity is bucketed in the declared symbol timezone, the same frame the
        resulting :class:`~pynecore.core.syminfo.SymInfoInterval` values are
        interpreted in; the host timezone must not leak into stored opening hours.

        :param candle: Newly committed real bar with a millisecond timestamp.
        """
        try:
            bar_datetime = datetime.fromtimestamp(
                candle.timestamp / 1000, tz=self._trading_hours_tz
            )
        except (OverflowError, OSError, ValueError):
            return
        key = (bar_datetime.isoweekday(), bar_datetime.hour)
        self._trading_hours[key] = self._trading_hours.get(key, 0) + 1

    def _collect_existing_trading_hours(self) -> None:
        """Sample existing v2 records for opening-hours analysis."""
        sample_interval = max(1, self._size // 1000)
        with _V2OHLCVReader(self.path) as reader:
            for position in range(0, self._size, sample_interval):
                self._collect_trading_hours(reader.read(position))

    def _has_enough_data_for_opening_hours(self) -> bool:
        """Return whether the declared period has enough activity for analysis.

        :return: Whether opening hours can be estimated.
        """
        if not self._trading_hours:
            return False
        if self._interval_unit >= 4:
            return len({day for day, _hour in self._trading_hours}) >= 3

        seconds_per_unit = {1: 1, 2: 60, 3: 3600}[self._interval_unit]
        interval_seconds = self._interval_value * seconds_per_unit
        data_points = sum(self._trading_hours.values())
        points_per_hour = 3600 / interval_seconds
        return data_points / points_per_hour >= 2.0

    def _analyze_opening_hours(self) -> None:
        """Infer weekly opening-hour intervals from collected activity."""
        if not self._trading_hours:
            self._analyzed_opening_hours = None
            return

        if self._interval_unit >= 4:
            hours: list[SymInfoInterval] = []
            days_with_trading = {day for day, _hour in self._trading_hours}
            if len(days_with_trading) == 7:
                for day in range(1, 8):
                    hours.append(
                        SymInfoInterval(day=day, start=time(0, 0, 0), end=time(23, 59, 59))
                    )
            elif days_with_trading <= {1, 2, 3, 4, 5}:
                for day in range(1, 6):
                    hours.append(
                        SymInfoInterval(day=day, start=time(9, 30, 0), end=time(16, 0, 0))
                    )
            else:
                for day in sorted(days_with_trading):
                    hours.append(
                        SymInfoInterval(day=day, start=time(0, 0, 0), end=time(23, 59, 59))
                    )
            self._analyzed_opening_hours = hours
            return

        if len(self._trading_hours) >= 168 * 0.7:
            counts = list(self._trading_hours.values())
            average_count = sum(counts) / len(counts)
            variance = sum((count - average_count) ** 2 for count in counts) / len(counts)
            if variance < average_count * 0.5:
                self._analyzed_opening_hours = [
                    SymInfoInterval(day=day, start=time(0, 0, 0), end=time(23, 59, 59))
                    for day in range(1, 8)
                ]
                return

        hours = []
        for day in range(1, 8):
            day_hours = [
                (hour, count)
                for (activity_day, hour), count in self._trading_hours.items()
                if activity_day == day
            ]
            if not day_hours:
                continue
            day_hours.sort(key=lambda item: item[0])
            total_count = sum(count for _hour, count in day_hours)
            if total_count == 0:
                continue
            threshold = total_count / len(day_hours) * 0.2
            periods: list[tuple[int, int]] = []
            current_start: int | None = None
            current_end: int | None = None
            for hour, count in day_hours:
                if count >= threshold:
                    if current_start is None:
                        current_start = hour
                    current_end = hour
                elif current_start is not None:
                    assert current_end is not None
                    periods.append((current_start, current_end))
                    current_start = None
                    current_end = None
            if current_start is not None:
                assert current_end is not None
                periods.append((current_start, current_end))

            for start_hour, end_hour in periods:
                hours.append(
                    SymInfoInterval(
                        day=day,
                        start=time(start_hour, 0, 0),
                        end=time(end_hour, 59, 59),
                    )
                )

        if not hours:
            hours = [
                SymInfoInterval(day=day, start=time(9, 30, 0), end=time(16, 0, 0))
                for day in range(1, 6)
            ]
        self._analyzed_opening_hours = hours

    def _set_columns(self, columns: tuple[_Column, ...]) -> None:
        self._columns = columns
        self._ordered_columns = tuple(sorted(columns, key=lambda column: column.byte_offset))
        self._record_struct = _record_struct(columns)
        self._header_size = _FIXED_HEADER_SIZE + len(columns) * _DESCRIPTOR.size
        self._record_size = self._record_struct.size

    def _write_empty_file(self) -> None:
        assert self._file is not None
        header = _build_header(
            self._columns,
            0,
            0,
            0,
            0,
            self._interval_value,
            self._interval_unit,
        )
        descriptors = b"".join(_pack_descriptor(column) for column in self._columns)
        metadata = header + descriptors
        if os.pwrite(self._file.fileno(), metadata, 0) != len(metadata):
            raise OSError("Could not write the complete OHLCV header and schema")
        self._file.truncate(self._header_size)
        os.fsync(self._file.fileno())

    @staticmethod
    def _validate_writer_schema(columns: tuple[_Column, ...]) -> None:
        roles = {column.role for column in columns}
        if roles != _REQUIRED_OHLCV_ROLES or len(columns) != len(_REQUIRED_OHLCV_ROLES):
            raise ValueError("OHLCV writer can append only to the standard OHLCV profile")
        by_role = {column.role: column for column in columns}
        expected_absolute = {
            _ROLE_TIMESTAMP: 2,
            _ROLE_OPEN: 6,
            _ROLE_VOLUME: 6,
        }
        for role, dtype in expected_absolute.items():
            column = by_role[role]
            if column.dtype != dtype or column.base != _ABSOLUTE_BASE:
                raise ValueError(f"Unsupported standard OHLCV schema for column {column.name!r}")
        for role in _DELTA_CANDIDATE_ROLES:
            column = by_role[role]
            valid_delta = column.dtype == 5 and column.base == _ROLE_OPEN
            valid_absolute = column.dtype == 6 and column.base == _ABSOLUTE_BASE
            if not (valid_delta or valid_absolute):
                raise ValueError(f"Unsupported standard OHLCV schema for column {column.name!r}")

        promoted_roles = frozenset(
            role for role in _DELTA_CANDIDATE_ROLES if by_role[role].dtype == 6
        )
        if columns != _make_default_columns(promoted_roles):
            raise ValueError("OHLCV schema does not match the packed standard profile layout")

    def _pack_record(self, candle: OHLCV) -> bytes:
        logical_values = {
            _ROLE_TIMESTAMP: candle.timestamp,
            _ROLE_OPEN: candle.open,
            _ROLE_HIGH: candle.high,
            _ROLE_LOW: candle.low,
            _ROLE_CLOSE: candle.close,
            _ROLE_VOLUME: candle.volume,
        }
        values: list[int | float] = []
        for column in self._ordered_columns:
            value = logical_values[column.role]
            if column.dtype == 2:
                values.append(value)
                continue
            float_value = float(value)
            if column.base != _ABSOLUTE_BASE:
                if math.isnan(float_value):
                    values.append(_CANONICAL_NAN32)
                    continue
                base_value = float(logical_values[column.base])
                if math.isnan(base_value):
                    raise ValueError(
                        f"OHLCV {column.name} cannot be delta-encoded from a NaN base"
                    )
                float_value -= base_value
                if not math.isfinite(float_value):
                    raise ValueError(
                        f"OHLCV {column.name} delta is outside the finite storage range"
                    )
            if math.isnan(float_value):
                values.append(_CANONICAL_NAN32 if column.dtype == 5 else _CANONICAL_NAN64)
            else:
                values.append(float_value)
        try:
            return self._record_struct.pack(*values)
        except OverflowError as error:
            raise ValueError("OHLCV record contains a value outside the configured storage range") from error

    def _dense_after_append(self, timestamp: int) -> bool:
        if self._interval_unit not in _FIXED_INTERVAL_MS or self._last_timestamp is None:
            return False
        expected = self._interval_value * _FIXED_INTERVAL_MS[self._interval_unit]
        boundary_matches = timestamp - self._last_timestamp == expected
        if self._size == 1:
            return boundary_matches
        if self._size >= 2:
            return self._dense and boundary_matches
        return False

    def _rebuild_with_append(self, candle: OHLCV, failed_roles: frozenset[int]) -> None:
        assert self._file is not None
        promoted_columns = _promote_columns(self._columns, failed_roles)
        path = Path(self.path)
        temp_name: str | None = None
        record_count = 0
        first_timestamp = 0
        last_timestamp = 0
        expected_interval = (
            self._interval_value * _FIXED_INTERVAL_MS[self._interval_unit]
            if self._interval_unit in _FIXED_INTERVAL_MS
            else None
        )
        dense = expected_interval is not None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                buffering=0,
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as replacement:
                temp_name = replacement.name
                record_struct = _record_struct(promoted_columns)
                old_columns = self._columns
                old_ordered = self._ordered_columns
                old_struct = self._record_struct
                try:
                    self._columns = promoted_columns
                    self._ordered_columns = tuple(
                        sorted(promoted_columns, key=lambda candidate: candidate.byte_offset)
                    )
                    self._record_struct = record_struct
                    provisional = _build_header(
                        promoted_columns,
                        0,
                        0,
                        0,
                        0,
                        self._interval_value,
                        self._interval_unit,
                    )
                    descriptors = b"".join(
                        _pack_descriptor(column) for column in promoted_columns
                    )
                    _write_all(
                        replacement,
                        provisional,
                        "Could not write the complete replacement OHLCV header",
                    )
                    _write_all(
                        replacement,
                        descriptors,
                        "Could not write the complete replacement OHLCV schema",
                    )
                    # The sidecar already holds the provisional row of the bar being
                    # appended, so it is one row ahead of the record count this copy
                    # reads. Only binary records are copied here — the sidecar is left
                    # untouched and becomes aligned again the moment the rebuilt file
                    # publishes the new record.
                    with _V2OHLCVReader(self.path, load_extra_fields=False) as reader:
                        for item in reader:
                            if record_count == 0:
                                first_timestamp = item.timestamp
                            elif expected_interval is None or (
                                item.timestamp - last_timestamp != expected_interval
                            ):
                                dense = False
                            _write_all(
                                replacement,
                                self._pack_record(item),
                                "Could not write a complete replacement OHLCV record",
                            )
                            last_timestamp = item.timestamp
                            record_count += 1

                    if record_count == 0:
                        first_timestamp = candle.timestamp
                    elif expected_interval is None or (
                        candle.timestamp - last_timestamp != expected_interval
                    ):
                        dense = False
                    _write_all(
                        replacement,
                        self._pack_record(candle),
                        "Could not write a complete replacement OHLCV record",
                    )
                    last_timestamp = candle.timestamp
                    record_count += 1
                    if record_count < 2:
                        dense = False
                    replacement.flush()
                    os.fsync(replacement.fileno())
                    final_header = _build_header(
                        promoted_columns,
                        _DENSE_FLAG if dense else 0,
                        record_count,
                        first_timestamp,
                        last_timestamp,
                        self._interval_value,
                        self._interval_unit,
                    )
                    if os.pwrite(replacement.fileno(), final_header, 0) != len(final_header):
                        raise OSError("Could not publish the complete replacement OHLCV header")
                    os.fsync(replacement.fileno())
                finally:
                    self._columns = old_columns
                    self._ordered_columns = old_ordered
                    self._record_struct = old_struct

            assert temp_name is not None
            os.replace(temp_name, path)
            temp_name = None
            self._rebuild_published = True
            try:
                replacement_file = open(path, "r+b", buffering=0)
            except Exception:
                self._file.close()
                self._file = None
                raise
            old_file = self._file
            self._file = replacement_file
            old_file.close()
            self._set_columns(promoted_columns)
            self._size = record_count
            self._start_timestamp = first_timestamp
            self._last_timestamp = last_timestamp
            self._dense = dense
            _fsync_directory(path.parent)
            self._collect_price_data(candle)
            self._collect_volume_data(candle)
            self._collect_trading_hours(candle)
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass


class _V2OHLCVReader:
    """Read a validated, immutable snapshot of an OHLCV v2 file through mmap.

    :param path: Source OHLCV v2 file path.
    """

    __slots__ = (
        "path",
        "_file",
        "_mmap",
        "_size",
        "_period",
        "_dense",
        "_start_timestamp",
        "_last_timestamp",
        "_header_size",
        "_record_size",
        "_columns",
        "_record_struct",
        "_role_indices",
        "_base_indices",
        "_base_roles",
        "_standard_ohlcv",
        "_timestamp_offset",
        "_timestamp_struct",
        "_extra_data",
        "_load_extra_fields",
    )

    def __init__(self, path: str | Path, *, load_extra_fields: bool = True):
        """Create a closed OHLCV v2 reader.

        Returned :class:`~pynecore.types.ohlcv.OHLCV` values carry Unix millisecond
        timestamps.

        :param path: Source OHLCV v2 file path.
        :param load_extra_fields: Whether to load and validate the position-aligned
            ``.extra.csv`` sidecar. The writer's schema rebuild turns this off: it
            copies binary records only, while the sidecar is deliberately one
            provisional row ahead of the still-unpublished record count.
        """
        self.path = str(path)
        self._load_extra_fields = load_extra_fields
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self._size = 0
        self._period = ""
        self._dense = False
        self._start_timestamp: int | None = None
        self._last_timestamp: int | None = None
        self._header_size = 0
        self._record_size = 0
        self._columns: tuple[_Column, ...] = ()
        self._record_struct = struct.Struct("<")
        self._role_indices: dict[int, int] = {}
        self._base_indices: dict[int, int] = {}
        self._base_roles: dict[int, int] = {}
        self._standard_ohlcv = False
        self._timestamp_offset = 0
        self._timestamp_struct = struct.Struct("<q")
        self._extra_data: list[dict[str, int | float | str]] | None = None

    def __enter__(self) -> "_V2OHLCVReader":
        """Open the reader and return it.

        :return: This reader.
        """
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the mmap and file on every context-manager exit path.

        :param exc_type: Exception type raised in the context, if any.
        :param exc_val: Exception value raised in the context, if any.
        :param exc_tb: Exception traceback raised in the context, if any.
        """
        self.close()

    @property
    def size(self) -> int:
        """Return the snapshot's authoritative committed record count.

        :return: Committed record count.
        """
        return self._size

    @property
    def period(self) -> str:
        """Return the header's canonical TradingView timeframe.

        :return: Declared file period.
        """
        return self._period

    @property
    def dense(self) -> bool:
        """Return the writer-verified DENSE flag from the header snapshot.

        :return: Whether every adjacent timestamp matches the fixed nominal interval.
        """
        return self._dense

    @property
    def start_timestamp(self) -> int | None:
        """Return the first committed timestamp in milliseconds.

        :return: First timestamp, or ``None`` for an empty file.
        """
        return self._start_timestamp

    @property
    def start_datetime(self) -> datetime:
        """Return the first committed timestamp as a UTC datetime.

        :return: First timestamp converted from milliseconds.
        :raises AssertionError: If the file is empty.
        """
        assert self._start_timestamp is not None
        return datetime.fromtimestamp(self._start_timestamp / 1000, UTC)

    @property
    def end_timestamp(self) -> int | None:
        """Return the actual last committed timestamp in milliseconds.

        :return: Last timestamp, or ``None`` for an empty file.
        """
        return self._last_timestamp

    @property
    def end_datetime(self) -> datetime:
        """Return the last committed timestamp as a UTC datetime.

        :return: Last timestamp converted from milliseconds.
        :raises AssertionError: If the file is empty.
        """
        assert self._last_timestamp is not None
        return datetime.fromtimestamp(self._last_timestamp / 1000, UTC)

    def open(self) -> "_V2OHLCVReader":
        """Validate the v2 header and schema, then mmap one committed snapshot.

        Bytes after ``header_size + record_count * record_size`` are ignored as an
        uncommitted tail. The reader never follows later file growth implicitly.

        :return: This reader.
        :raises ValueError: If the v2 header, schema, or committed extent is invalid.
        """
        if self._file is not None:
            return self
        file = open(self.path, "rb", buffering=0)
        return self.open_file(file)

    def open_file(self, file: BinaryIO, magic: bytes | None = None) -> "_V2OHLCVReader":
        """Adopt an open file and expose its committed v2 snapshot.

        :param file: Open binary file positioned arbitrarily.
        :param magic: Already-read eight-byte magic, or ``None`` to read the full header.
        :return: This reader.
        :raises ValueError: If the v2 header, schema, or committed extent is invalid.
        """
        self._file = file
        try:
            file_size = os.fstat(file.fileno()).st_size
            layout = _read_layout(file, file_size, magic)
            roles = {column.role for column in layout.columns}
            if not _REQUIRED_OHLCV_ROLES.issubset(roles):
                raise ValueError("OHLCV schema lacks one or more required OHLCV columns")

            ordered = tuple(sorted(layout.columns, key=lambda column: column.byte_offset))
            self._record_struct = _record_struct(layout.columns)
            self._role_indices = {column.role: index for index, column in enumerate(ordered)}
            self._base_indices = {
                column.role: (
                    -1 if column.base == _ABSOLUTE_BASE else self._role_indices[column.base]
                )
                for column in ordered
            }
            self._base_roles = {column.role: column.base for column in ordered}
            by_role = {column.role: column for column in ordered}
            self._standard_ohlcv = (
                by_role[_ROLE_OPEN].base == _ABSOLUTE_BASE
                and by_role[_ROLE_VOLUME].base == _ABSOLUTE_BASE
                and all(
                    by_role[role].base in (_ABSOLUTE_BASE, _ROLE_OPEN)
                    for role in _DELTA_CANDIDATE_ROLES
                )
            )
            timestamp_column = next(
                column for column in layout.columns if column.role == _ROLE_TIMESTAMP
            )
            self._timestamp_offset = timestamp_column.byte_offset
            self._header_size = layout.header_size
            self._record_size = layout.record_size
            self._columns = layout.columns
            self._size = layout.record_count
            self._period = _period_from_interval(layout.interval_value, layout.interval_unit)
            self._dense = bool(layout.flags & _DENSE_FLAG)
            self._start_timestamp = layout.first_timestamp if layout.record_count else None
            self._last_timestamp = layout.last_timestamp if layout.record_count else None
            self._load_extra_csv()
            committed_end = layout.header_size + layout.record_count * layout.record_size
            self._mmap = mmap.mmap(
                file.fileno(), committed_end, access=mmap.ACCESS_READ
            )
        except Exception:
            file.close()
            self._file = None
            raise
        return self

    def _load_extra_csv(self) -> None:
        """Load the v2 position-aligned user extra-field sidecar once.

        Sidecar data row ``N`` corresponds exactly to committed v2 record ``N``.
        Because v2 stores no phantom gap records, missing time intervals create no
        binary record and no sidecar row. A present sidecar must therefore contain
        exactly :attr:`size` data rows; mismatches are rejected instead of silently
        attaching values to the wrong bars.

        Without a sidecar — or when the reader was created with
        ``load_extra_fields=False`` — :class:`OHLCV` values retain their default
        ``extra_fields=None``.

        :raises ValueError: If the sidecar shape does not match the v2 snapshot.
        """
        if not self._load_extra_fields:
            self._extra_data = None
            return
        extra_path = _extra_sidecar_path(self.path)
        if not extra_path.exists():
            self._extra_data = None
            return

        with open(extra_path, "r", encoding="utf-8-sig", newline="") as extra_file:
            reader = csv.reader(extra_file)
            headers = next(reader, None)
            if not headers:
                self._extra_data = None
                return

            rows: list[list[str]] = []
            numeric_columns: list[bool | None] = [None] * len(headers)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise ValueError(
                        f"OHLCV extra sidecar row {row_number} has {len(row)} columns; "
                        f"expected {len(headers)}"
                    )
                rows.append(row)
                for index, value in enumerate(row):
                    if numeric_columns[index] is not None or not value:
                        continue
                    if value.lower() in ("nan", "na"):
                        continue
                    try:
                        float(value)
                    except ValueError:
                        numeric_columns[index] = False
                    else:
                        numeric_columns[index] = True

        if len(rows) != self._size:
            raise ValueError(
                f"OHLCV extra sidecar has {len(rows)} data rows; expected {self._size} "
                "for position alignment"
            )

        resolved_numeric = [value if value is not None else False for value in numeric_columns]
        extra_data: list[dict[str, int | float | str]] = []
        for row in rows:
            parsed: dict[str, int | float | str] = {}
            for index, header in enumerate(headers):
                value = row[index]
                if resolved_numeric[index]:
                    parsed[header] = (
                        float("nan") if not value or value.lower() in ("nan", "na") else float(value)
                    )
                else:
                    parsed[header] = value
            extra_data.append(parsed)
        self._extra_data = extra_data

    def close(self) -> None:
        """Close the mmap and file; repeated calls are safe."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._extra_data = None

    def __iter__(self) -> Iterator[OHLCV]:
        """Iterate over every committed real record exactly once.

        :return: Iterator of OHLCV values whose timestamps are milliseconds.
        """
        for position in range(self._size):
            yield self.read(position)

    def read(self, position: int) -> OHLCV:
        """Read one zero-based committed record by mmap offset.

        :param position: Zero-based committed record index.
        :return: Decoded OHLCV value with a millisecond timestamp.
        :raises IndexError: If ``position`` is outside the committed snapshot.
        :raises RuntimeError: If the reader is closed.
        """
        if position < 0 or position >= self._size:
            raise IndexError("OHLCV position out of range")
        if self._mmap is None:
            raise RuntimeError("OHLCV reader is not open")

        offset = self._header_size + position * self._record_size
        values = self._record_struct.unpack_from(self._mmap, offset)
        role_indices = self._role_indices
        base_indices = self._base_indices

        timestamp = values[role_indices[_ROLE_TIMESTAMP]]
        extra_fields = None if self._extra_data is None else self._extra_data[position]
        if not self._standard_ohlcv:
            return OHLCV(
                int(timestamp),
                self._decode_value(values, _ROLE_OPEN),
                self._decode_value(values, _ROLE_HIGH),
                self._decode_value(values, _ROLE_LOW),
                self._decode_value(values, _ROLE_CLOSE),
                self._decode_value(values, _ROLE_VOLUME),
                extra_fields=extra_fields,
            )

        open_value = values[role_indices[_ROLE_OPEN]]

        high = values[role_indices[_ROLE_HIGH]]
        high_base_index = base_indices[_ROLE_HIGH]
        if high_base_index >= 0:
            high_base = values[high_base_index]
            high = math.nan if math.isnan(high) or math.isnan(high_base) else high_base + high

        low = values[role_indices[_ROLE_LOW]]
        low_base_index = base_indices[_ROLE_LOW]
        if low_base_index >= 0:
            low_base = values[low_base_index]
            low = math.nan if math.isnan(low) or math.isnan(low_base) else low_base + low

        close = values[role_indices[_ROLE_CLOSE]]
        close_base_index = base_indices[_ROLE_CLOSE]
        if close_base_index >= 0:
            close_base = values[close_base_index]
            close = math.nan if math.isnan(close) or math.isnan(close_base) else close_base + close

        volume = values[role_indices[_ROLE_VOLUME]]
        volume_base_index = base_indices[_ROLE_VOLUME]
        if volume_base_index >= 0:
            volume_base = values[volume_base_index]
            volume = (
                math.nan
                if math.isnan(volume) or math.isnan(volume_base)
                else volume_base + volume
            )

        return OHLCV(
            int(timestamp),
            open_value,
            high,
            low,
            close,
            volume,
            extra_fields=extra_fields,
        )

    def _decode_value(self, values: tuple[int | float, ...], role: int) -> float:
        value = float(values[self._role_indices[role]])
        base = self._base_roles[role]
        while base != _ABSOLUTE_BASE:
            if math.isnan(value):
                return math.nan
            base_value = float(values[self._role_indices[base]])
            if math.isnan(base_value):
                return math.nan
            value += base_value
            base = self._base_roles[base]
        return value

    def read_from(
        self, start_timestamp: int, end_timestamp: int | None = None
    ) -> Iterator[OHLCV]:
        """Iterate over an inclusive millisecond timestamp window.

        Exactly one lower-bound and, when needed, one upper-bound bisect run before
        iteration. Nominal interval metadata is never used for addressing.

        :param start_timestamp: Inclusive lower timestamp bound in milliseconds.
        :param end_timestamp: Inclusive upper bound, or ``None`` for the file end.
        :return: Iterator over committed bars in the requested window.
        """
        start_position = self._bisect_left(start_timestamp)
        end_position = self._size if end_timestamp is None else self._bisect_right(end_timestamp)
        for position in range(start_position, end_position):
            yield self.read(position)

    def get_positions(
        self, start_timestamp: int | None = None, end_timestamp: int | None = None
    ) -> tuple[int, int]:
        """Return bisected bounds for an inclusive millisecond timestamp window.

        :param start_timestamp: Inclusive lower bound, or ``None`` for index zero.
        :param end_timestamp: Inclusive upper bound, or ``None`` for the snapshot end.
        :return: ``(start_index, end_index_exclusive)``.
        """
        start_position = 0 if start_timestamp is None else self._bisect_left(start_timestamp)
        end_position = self._size if end_timestamp is None else self._bisect_right(end_timestamp)
        return start_position, max(start_position, end_position)

    def get_size(
        self, start_timestamp: int | None = None, end_timestamp: int | None = None
    ) -> int:
        """Return the committed record count in an inclusive timestamp window.

        :param start_timestamp: Inclusive lower bound in milliseconds.
        :param end_timestamp: Inclusive upper bound in milliseconds.
        :return: Number of records in the window.
        """
        start_position, end_position = self.get_positions(start_timestamp, end_timestamp)
        return end_position - start_position

    def save_to_csv(self, path: str | Path, as_datetime: bool = False) -> None:
        """Export every committed record to CSV.

        :param path: Destination CSV path.
        :param as_datetime: Emit UTC datetimes instead of integer millisecond timestamps.
        """
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("time" if as_datetime else "timestamp", "open", "high", "low", "close", "volume"))
            for candle in self:
                timestamp: int | datetime
                if as_datetime:
                    timestamp = datetime.fromtimestamp(candle.timestamp / 1000, UTC)
                else:
                    timestamp = candle.timestamp
                writer.writerow(
                    (timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume)
                )

    def save_to_json(self, path: str | Path, as_datetime: bool = False) -> None:
        """Export every committed record to JSON.

        :param path: Destination JSON path.
        :param as_datetime: Emit UTC ISO datetimes instead of integer millisecond timestamps.
        """
        data: list[dict[str, int | float | str]] = []
        for candle in self:
            timestamp_key = "time" if as_datetime else "timestamp"
            timestamp: int | str
            if as_datetime:
                timestamp = datetime.fromtimestamp(candle.timestamp / 1000, UTC).isoformat()
            else:
                timestamp = candle.timestamp
            data.append(
                {
                    timestamp_key: timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
        with open(path, "w") as file:
            json.dump(data, file, indent=2, allow_nan=True)

    def _timestamp_at(self, position: int) -> int:
        assert self._mmap is not None
        offset = self._header_size + position * self._record_size + self._timestamp_offset
        return self._timestamp_struct.unpack_from(self._mmap, offset)[0]

    def _bisect_left(self, timestamp: int) -> int:
        low = 0
        high = self._size
        while low < high:
            middle = (low + high) // 2
            if self._timestamp_at(middle) < timestamp:
                low = middle + 1
            else:
                high = middle
        return low

    def _bisect_right(self, timestamp: int) -> int:
        low = 0
        high = self._size
        while low < high:
            middle = (low + high) // 2
            if timestamp < self._timestamp_at(middle):
                high = middle
            else:
                low = middle + 1
        return low


class OHLCVReader:
    """Read v2 and legacy v1 OHLCV files through one public interface.

    :class:`OHLCV` values and timestamp properties always use Unix milliseconds,
    regardless of the on-disk format. Format selection happens once in :meth:`open`;
    all data-path methods delegate directly to the selected concrete reader and do
    not test the file version.

    Legacy v1 files do not declare a period and do not carry a writer-verified DENSE
    flag. Consequently, :attr:`period` and :attr:`dense` return ``None`` for v1
    files rather than inferring or fabricating metadata.

    :param path: Source OHLCV file path.
    """

    __slots__ = ("path", "_reader", "_opened", "_period", "_dense")

    def __init__(self, path: str | Path):
        """Create a closed format-transparent reader.

        :param path: Source OHLCV file path.
        """
        self.path = str(path)
        self._reader: _V2OHLCVReader | _LegacyOHLCVReader = _V2OHLCVReader(path)
        self._opened = False
        self._period: str | None = None
        self._dense: bool | None = None

    def __enter__(self) -> "OHLCVReader":
        """Open the reader and return it.

        :return: This reader.
        """
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the selected reader on every context-manager exit path.

        :param exc_type: Exception type raised in the context, if any.
        :param exc_val: Exception value raised in the context, if any.
        :param exc_tb: Exception traceback raised in the context, if any.
        """
        self.close()

    @property
    def size(self) -> int:
        """Return the selected reader's record count.

        :return: Number of records.
        """
        return self._reader.size

    @property
    def period(self) -> str | None:
        """Return the declared v2 period, or ``None`` for legacy v1 files.

        :return: Canonical v2 period, or ``None`` when the file does not declare one.
        """
        return self._period

    @property
    def dense(self) -> bool | None:
        """Return the verified v2 DENSE flag, or ``None`` for legacy v1 files.

        :return: Verified density, or ``None`` when the file has no DENSE metadata.
        """
        return self._dense

    @property
    def start_timestamp(self) -> int | None:
        """Return the first timestamp in milliseconds.

        :return: First timestamp, or ``None`` for an empty file.
        """
        return self._reader.start_timestamp

    @property
    def end_timestamp(self) -> int | None:
        """Return the last timestamp in milliseconds.

        :return: Last timestamp, or ``None`` for an empty file.
        """
        return self._reader.end_timestamp

    @property
    def start_datetime(self) -> datetime:
        """Return the first timestamp as a UTC datetime.

        :return: First timestamp converted from milliseconds.
        """
        return self._reader.start_datetime

    @property
    def end_datetime(self) -> datetime:
        """Return the last timestamp as a UTC datetime.

        :return: Last timestamp converted from milliseconds.
        """
        return self._reader.end_datetime

    def open(self) -> "OHLCVReader":
        """Select the on-disk format once and open its concrete reader.

        The first eight bytes are the sole format discriminator. The v2 magic selects
        the native reader; every other prefix selects the read-only legacy reader.

        :return: This reader.
        """
        if self._opened:
            return self

        file = open(self.path, "rb", buffering=0)
        try:
            magic = file.read(len(_MAGIC))
        except Exception:
            file.close()
            raise

        if magic == _MAGIC:
            native_reader = _V2OHLCVReader(self.path)
            native_reader.open_file(file, magic)
            reader: _V2OHLCVReader | _LegacyOHLCVReader = native_reader
            period: str | None = native_reader.period
            dense: bool | None = native_reader.dense
        else:
            file.close()
            legacy_reader = _LegacyOHLCVReader(self.path)
            legacy_reader.open()
            reader = legacy_reader
            period = None
            dense = None

        self._reader = reader
        self._period = period
        self._dense = dense
        self._opened = True
        return self

    def close(self) -> None:
        """Close the selected concrete reader; repeated calls are safe."""
        self._reader.close()
        self._opened = False

    def __iter__(self) -> Iterator[OHLCV]:
        """Iterate over all records without testing the file version.

        :return: Iterator of OHLCV values with millisecond timestamps.
        """
        return iter(self._reader)

    def read(self, position: int) -> OHLCV:
        """Read one record without testing the file version.

        :param position: Zero-based record index.
        :return: OHLCV value with a millisecond timestamp.
        """
        return self._reader.read(position)

    def read_from(
        self, start_timestamp: int, end_timestamp: int | None = None
    ) -> Iterator[OHLCV]:
        """Read an inclusive millisecond timestamp range.

        :param start_timestamp: Inclusive lower timestamp bound in milliseconds.
        :param end_timestamp: Inclusive upper bound, or ``None`` for the file end.
        :return: Iterator over matching OHLCV values.
        """
        return self._reader.read_from(start_timestamp, end_timestamp)

    def get_positions(
        self, start_timestamp: int | None = None, end_timestamp: int | None = None
    ) -> tuple[int, int]:
        """Return positions for an inclusive millisecond timestamp range.

        :param start_timestamp: Inclusive lower bound, or ``None`` for the beginning.
        :param end_timestamp: Inclusive upper bound, or ``None`` for the end.
        :return: Half-open record position range.
        """
        return self._reader.get_positions(start_timestamp, end_timestamp)

    def get_size(
        self, start_timestamp: int | None = None, end_timestamp: int | None = None
    ) -> int:
        """Return the record count in a millisecond timestamp range.

        :param start_timestamp: Inclusive lower bound, or ``None`` for the beginning.
        :param end_timestamp: Inclusive upper bound, or ``None`` for the end.
        :return: Number of matching records.
        """
        return self._reader.get_size(start_timestamp, end_timestamp)

    def save_to_csv(self, path: str | Path, as_datetime: bool = False) -> None:
        """Export all records to CSV without testing the file version.

        :param path: Destination CSV path.
        :param as_datetime: Emit UTC datetimes instead of millisecond timestamps.
        """
        self._reader.save_to_csv(str(path), as_datetime)

    def save_to_json(self, path: str | Path, as_datetime: bool = False) -> None:
        """Export all records to JSON without testing the file version.

        :param path: Destination JSON path.
        :param as_datetime: Emit UTC datetimes instead of millisecond timestamps.
        """
        self._reader.save_to_json(str(path), as_datetime)
