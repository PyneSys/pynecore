"""Import CSV, TXT, and JSON market data into an open OHLCV v2 writer."""

import csv
import json
import os
import re
import tempfile
from calendar import monthrange
from collections.abc import Iterable, Iterator, Mapping
from math import gcd
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol, cast

from pynecore.core.ohlcv import OHLCVWriter, parse_timezone_name
from pynecore.types.ohlcv import OHLCV

__all__ = [
    "infer_csv_period",
    "infer_json_period",
    "infer_txt_period",
    "load_from_csv",
    "load_from_json",
    "load_from_txt",
]

_TIMESTAMP_NAMES = ("timestamp", "time", "date", "datetime", "ts_event", "ts_recv")
_STANDARD_COLUMNS = frozenset((*_TIMESTAMP_NAMES, "open", "high", "low", "close", "volume"))
_JSON_WRAPPER_NAMES = ("data", "candles", "ohlcv", "results")
_SUB_MICROSECONDS = re.compile(r"(\.\d{6})\d+")

_Timezone = tzinfo
_JsonNumberInput = str | int | float


class _RowWriter(Protocol):
    def writerow(self, row: Iterable[object], /) -> object:
        """Write one CSV row."""


class _ExtraSidecar:
    """Build a sidecar atomically while accepted bars are written.

    A sidecar data row is written only after the matching OHLCV bar has been accepted
    by the v2 writer. Data row ``N`` therefore corresponds exactly to binary record
    ``N``. Missing time intervals produce neither binary records nor empty sidecar rows.

    :param path: Final sidecar path, or ``None`` to disable sidecar output.
    :param headers: Extra-column headers in source order.
    """

    def __init__(self, path: str | Path | None, headers: list[str]):
        self._path = Path(path) if path is not None else None
        self._headers = headers
        self._file: IO[str] | None = None
        self._writer: _RowWriter | None = None
        self._temp_path: Path | None = None

    def __enter__(self) -> "_ExtraSidecar":
        if self._path is None or not self._headers:
            return self

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._file = temp_file
        self._temp_path = Path(temp_file.name)
        row_writer = csv.writer(temp_file)
        self._writer = row_writer
        row_writer.writerow(self._headers)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()

        if exc_type is not None:
            if self._temp_path is not None:
                self._temp_path.unlink(missing_ok=True)
            return

        if self._path is None:
            return
        if not self._headers:
            self._path.unlink(missing_ok=True)
            return

        assert self._temp_path is not None
        os.replace(self._temp_path, self._path)
        self._temp_path = None

    def write(self, row: list[str]) -> None:
        """Write one source-aligned extra-field row.

        :param row: Extra values in header order.
        """
        if self._writer is not None:
            self._writer.writerow(row)


def _parse_timezone_param(tz: str | None) -> _Timezone | None:
    """Parse a timezone name or numeric UTC offset.

    :param tz: Timezone name such as ``UTC`` or offset such as ``+01:00``.
    :return: Parsed timezone, or ``None`` when no timezone was supplied.
    :raises ValueError: If the timezone cannot be parsed.
    """
    if not tz:
        return None
    return parse_timezone_name(tz)


def _find_timestamp_columns(
    headers: list[str],
    timestamp_column: str | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
) -> tuple[int | None, int | None, int | None]:
    """Find timestamp-related column indices.

    :param headers: Normalized lowercase headers.
    :param timestamp_column: Explicit combined timestamp column.
    :param date_column: Explicit date column for split timestamps.
    :param time_column: Explicit time column for split timestamps.
    :return: ``(timestamp_index, date_index, time_index)``.
    :raises ValueError: If the requested or automatic timestamp columns are missing.
    """
    if (date_column is None) != (time_column is None):
        raise ValueError("date_column and time_column must be supplied together")

    if date_column is not None and time_column is not None:
        try:
            return None, headers.index(date_column.lower()), headers.index(time_column.lower())
        except ValueError as error:
            raise ValueError(
                f"Date/time columns not found: {date_column}/{time_column}"
            ) from error

    if timestamp_column is not None:
        normalized = timestamp_column.lower()
        try:
            return headers.index(normalized), None, None
        except ValueError as error:
            raise ValueError(f"Timestamp column not found: {normalized}") from error

    for name in _TIMESTAMP_NAMES:
        try:
            return headers.index(name), None, None
        except ValueError:
            continue
    raise ValueError("Timestamp column not found")


def _find_ohlcv_columns(headers: list[str]) -> tuple[int, int, int, int, int]:
    """Find the five required price and volume column indices.

    :param headers: Normalized lowercase headers.
    :return: ``(open, high, low, close, volume)`` indices.
    :raises ValueError: If a required column is missing.
    """
    indices: list[int] = []
    for name in ("open", "high", "low", "close", "volume"):
        try:
            indices.append(headers.index(name))
        except ValueError as error:
            raise ValueError(f"Missing required column: {name}") from error
    return indices[0], indices[1], indices[2], indices[3], indices[4]


def _parse_timestamp(
    timestamp_text: str,
    timestamp_format: str | None = None,
    timezone: _Timezone | None = None,
    *,
    wall_clock: bool = False,
) -> int:
    """Parse a source timestamp into Unix milliseconds.

    Numeric inputs are classified by their decimal width: up to 10 digits are
    seconds, 11-13 are milliseconds, 14-16 are microseconds, and 17-19 are
    nanoseconds. Datetime inputs retain an embedded timezone; otherwise the supplied
    timezone is applied. A trailing ``Z`` always means UTC.

    :param timestamp_text: Timestamp text to parse.
    :param timestamp_format: Explicit ``datetime.strptime`` format, if needed.
    :param timezone: Timezone for otherwise-naive datetimes.
    :param wall_clock: Ignore timezone offsets for cadence inference.
    :return: Unix or wall-clock timestamp in milliseconds.
    :raises ValueError: If the timestamp cannot be parsed.
    """
    value = timestamp_text.strip()
    if re.fullmatch(r"[+-]?\d+", value):
        timestamp = int(value)
        sign = -1 if timestamp < 0 else 1
        magnitude = abs(timestamp)
        digits = len(str(magnitude))
        if digits <= 10:
            unit = "s"
        elif digits <= 13:
            unit = "ms"
        elif digits <= 16:
            unit = "us"
        elif digits <= 19:
            unit = "ns"
        else:
            raise ValueError(
                "Numeric timestamp has more than 19 digits and cannot be interpreted"
            )
        if unit == "s":
            magnitude *= 1000
        elif unit == "us":
            magnitude //= 1000
        elif unit == "ns":
            magnitude //= 1_000_000
        return sign * magnitude

    value = _SUB_MICROSECONDS.sub(r"\1", value)
    has_utc_suffix = value.endswith("Z")
    parsed: datetime | None = None
    if timestamp_format is not None:
        parsed = datetime.strptime(value, timestamp_format)
    else:
        for candidate in (
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S%Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M",
            "%Y%m%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(value, candidate)
                break
            except ValueError:
                continue

    if parsed is None:
        raise ValueError(f"Could not parse timestamp: {timestamp_text}")
    parsed_datetime: datetime = parsed
    if parsed_datetime.tzinfo is None:
        if has_utc_suffix:
            parsed_datetime = parsed_datetime.replace(tzinfo=UTC)
        elif timezone is not None:
            parsed_datetime = parsed_datetime.replace(tzinfo=timezone)

    if wall_clock:
        wall_datetime = parsed_datetime.replace(tzinfo=None)
        wall_delta = wall_datetime - datetime(1970, 1, 1)
        return (
            (wall_delta.days * 86_400 + wall_delta.seconds) * 1000
            + wall_delta.microseconds // 1000
        )

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.astimezone()
    utc_datetime = parsed_datetime.astimezone(UTC)
    epoch_delta = utc_datetime - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        (epoch_delta.days * 86_400 + epoch_delta.seconds) * 1000
        + epoch_delta.microseconds // 1000
    )


def _timestamp_text(
    row: list[str],
    timestamp_index: int | None,
    date_index: int | None,
    time_index: int | None,
) -> str:
    if date_index is not None and time_index is not None:
        return f"{row[date_index]} {row[time_index]}"
    assert timestamp_index is not None
    return row[timestamp_index]


def _extra_columns(raw_headers: list[str], normalized_headers: list[str]) -> tuple[list[int], list[str]]:
    indices = [
        index for index, header in enumerate(normalized_headers) if header not in _STANDARD_COLUMNS
    ]
    return indices, [raw_headers[index].strip() for index in indices]


def _parse_and_write_ohlcv_row(
    writer: OHLCVWriter,
    source_path: Path,
    row_number: int,
    timestamp_text: str,
    row: list[str],
    ohlcv_indices: tuple[int, int, int, int, int],
    timestamp_format: str | None,
    timezone: _Timezone | None,
) -> None:
    """Parse, validate, and write one delimited source row.

    :param writer: Open v2 writer.
    :param source_path: Source path used in diagnostics.
    :param row_number: One-based source row number.
    :param timestamp_text: Combined or direct timestamp text.
    :param row: Full source row.
    :param ohlcv_indices: Price and volume column indices.
    :param timestamp_format: Explicit timestamp format, if any.
    :param timezone: Timezone for naive timestamps.
    :raises ValueError: If the row cannot produce a valid v2 record.
    """
    try:
        timestamp = _parse_timestamp(timestamp_text, timestamp_format, timezone)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{source_path}: row {row_number}: failed to parse timestamp "
            f"{timestamp_text!r}: {error}"
        ) from error

    try:
        open_index, high_index, low_index, close_index, volume_index = ohlcv_indices
        candle = OHLCV(
            timestamp,
            float(row[open_index]),
            float(row[high_index]),
            float(row[low_index]),
            float(row[close_index]),
            float(row[volume_index]),
        )
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"{source_path}: row {row_number}: invalid OHLCV values: {error}") from error

    try:
        writer.write(candle)
    except ValueError as error:
        raise ValueError(f"{source_path}: row {row_number}: {error}") from error


def _calendar_month_step(previous_wall_clock: int, wall_clock: int) -> int | None:
    """Return how many whole calendar months separate two wall-clock instants.

    A pair is month-aligned when both instants share the time of day and either the
    same day of month (bars stamped with their opening day) or the last day of their
    own month (bars stamped with their closing day, where January 31 is followed by
    February 28). Anything else belongs to a fixed-duration cadence — a four-week
    series, for instance — that must not be labelled ``M``.

    :param previous_wall_clock: Earlier wall-clock timestamp in milliseconds.
    :param wall_clock: Later wall-clock timestamp in milliseconds.
    :return: Positive month step, or ``None`` if the pair is not month-aligned.
    """
    epoch = datetime(1970, 1, 1)
    start = epoch + timedelta(milliseconds=previous_wall_clock)
    end = epoch + timedelta(milliseconds=wall_clock)
    if start.time() != end.time():
        return None
    if start.day != end.day:
        start_is_month_end = start.day == monthrange(start.year, start.month)[1]
        end_is_month_end = end.day == monthrange(end.year, end.month)[1]
        if not (start_is_month_end and end_is_month_end):
            return None
    step = (end.year - start.year) * 12 + end.month - start.month
    return step if step > 0 else None


def _infer_period(timestamps: Iterator[tuple[str, int, int]]) -> str:
    """Derive the nominal v2 period from the smallest positive source interval.

    Each item is ``(location, utc_timestamp_ms, wall_clock_timestamp_ms)``. The
    cadence is measured on the wall clock so a daylight-saving transition — which
    stretches or shrinks the UTC interval of an otherwise regular series — does not
    masquerade as a shorter period. The UTC value stays authoritative for ordering
    and duplicate detection, and is used as the interval whenever the wall-clock
    delta is not positive (a backwards local-clock shift).

    Calendar months are recognised from the wall-clock calendar fields before the
    elapsed-duration ladder is consulted, because months have no fixed length: a
    monthly series covering a leap February would otherwise be declared ``29D`` and
    one that never spans a February ``30D`` or ``31D``. Missing bars only widen the
    month steps, so the nominal period is their greatest common divisor rather than
    a single repeated step.

    :param timestamps: Located timestamp pairs in source order.
    :return: Canonical TradingView period for the v2 header.
    :raises ValueError: If timestamps are duplicated, unordered, or too few.
    """
    previous: int | None = None
    previous_wall_clock: int | None = None
    smallest_interval: int | None = None
    month_step: int | None = None
    calendar_monthly = True
    count = 0

    for location, timestamp, wall_clock in timestamps:
        count += 1
        if previous is not None and previous_wall_clock is not None:
            elapsed_interval = timestamp - previous
            if elapsed_interval == 0:
                raise ValueError(f"{location}: duplicate timestamp {timestamp}")
            if elapsed_interval < 0:
                raise ValueError(
                    f"{location}: timestamps must be strictly increasing; "
                    f"{timestamp} follows {previous}"
                )
            wall_interval = wall_clock - previous_wall_clock
            interval = wall_interval if wall_interval > 0 else elapsed_interval
            if smallest_interval is None or interval < smallest_interval:
                smallest_interval = interval
            if calendar_monthly:
                step = _calendar_month_step(previous_wall_clock, wall_clock)
                if step is None:
                    calendar_monthly = False
                else:
                    month_step = step if month_step is None else gcd(month_step, step)
        previous = timestamp
        previous_wall_clock = wall_clock

    if count < 2 or smallest_interval is None:
        raise ValueError("Cannot infer OHLCV period from fewer than two records")
    if calendar_monthly and month_step is not None:
        return f"{month_step}M"
    if smallest_interval % 1000 != 0:
        raise ValueError(
            f"Cannot represent inferred {smallest_interval} ms period in the OHLCV v2 header"
        )

    seconds = smallest_interval // 1000
    week = 60 * 60 * 24 * 7
    day = 60 * 60 * 24
    # No month heuristic below this point. Calendar months are recognised above from
    # the wall-clock fields, gaps included; anything that reaches here is measured by
    # elapsed duration alone, so a 28-day step is four weeks. Reading it as a month
    # would mislabel every four-week feed that happens to be missing a bar, whose
    # steps are then 28 and 56 days long.
    if seconds % week == 0:
        return f"{seconds // week}W"
    if seconds % day == 0:
        return f"{seconds // day}D"
    if seconds % 60 == 0:
        return str(seconds // 60)
    return f"{seconds}S"


def _csv_timestamps(
    path: Path,
    timestamp_format: str | None,
    timestamp_column: str | None,
    date_column: str | None,
    time_column: str | None,
    timezone: _Timezone | None,
) -> Iterator[tuple[str, int, int]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        raw_headers = next(reader, None)
        if not raw_headers:
            raise ValueError(f"{path}: CSV header row is missing")
        headers = [header.strip().lower() for header in raw_headers]
        timestamp_indices = _find_timestamp_columns(
            headers, timestamp_column, date_column, time_column
        )
        for row in reader:
            row_number = reader.line_num
            try:
                text = _timestamp_text(row, *timestamp_indices)
                timestamp = _parse_timestamp(text, timestamp_format, timezone)
                wall_clock = _parse_timestamp(
                    text, timestamp_format, timezone, wall_clock=True
                )
            except (IndexError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{path}: row {row_number}: invalid timestamp: {error}") from error
            yield f"{path}: row {row_number}", timestamp, wall_clock


def infer_csv_period(
    path: str | Path,
    timestamp_format: str | None = None,
    timestamp_column: str | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    tz: str | None = None,
) -> str:
    """Infer the nominal v2 period from the smallest positive CSV timestamp delta.

    :param path: Source CSV path.
    :param timestamp_format: Explicit timestamp format, if needed.
    :param timestamp_column: Explicit combined timestamp column.
    :param date_column: Explicit date column for split timestamps.
    :param time_column: Explicit time column for split timestamps.
    :param tz: Timezone for naive timestamps.
    :return: Canonical TradingView period for the v2 header.
    :raises ValueError: If timestamps are malformed, unordered, or insufficient.
    """
    source_path = Path(path)
    timezone = _parse_timezone_param(tz)
    return _infer_period(
        _csv_timestamps(
            source_path,
            timestamp_format,
            timestamp_column,
            date_column,
            time_column,
            timezone,
        )
    )


def load_from_csv(
    writer: OHLCVWriter,
    path: str | Path,
    timestamp_format: str | None = None,
    timestamp_column: str | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    tz: str | None = None,
    extra_csv_path: str | Path | None = None,
) -> None:
    """Load CSV rows into an open v2 writer without filling timestamp gaps.

    When ``extra_csv_path`` is supplied, non-OHLCV source columns are written to an
    atomic sidecar. Its data rows are strictly 1:1 and position-aligned with accepted
    binary records; absent timestamps create no placeholder rows.

    :param writer: Open v2 writer.
    :param path: Source CSV path.
    :param timestamp_format: Explicit timestamp format, if needed.
    :param timestamp_column: Explicit combined timestamp column.
    :param date_column: Explicit date column for split timestamps.
    :param time_column: Explicit time column for split timestamps.
    :param tz: Timezone for naive timestamps.
    :param extra_csv_path: Optional destination for user extra columns.
    :raises ValueError: If headers or any source row are malformed.
    """
    source_path = Path(path)
    timezone = _parse_timezone_param(tz)

    with open(source_path, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        raw_headers = next(reader, None)
        if not raw_headers:
            raise ValueError(f"{source_path}: CSV header row is missing")
        headers = [header.strip().lower() for header in raw_headers]
        timestamp_indices = _find_timestamp_columns(
            headers, timestamp_column, date_column, time_column
        )
        ohlcv_indices = _find_ohlcv_columns(headers)
        extra_indices, extra_headers = _extra_columns(raw_headers, headers)

        with _ExtraSidecar(extra_csv_path, extra_headers) as sidecar:
            for row in reader:
                row_number = reader.line_num
                try:
                    timestamp_text = _timestamp_text(row, *timestamp_indices)
                except IndexError as error:
                    raise ValueError(
                        f"{source_path}: row {row_number}: timestamp column is missing"
                    ) from error
                _parse_and_write_ohlcv_row(
                    writer,
                    source_path,
                    row_number,
                    timestamp_text,
                    row,
                    ohlcv_indices,
                    timestamp_format,
                    timezone,
                )
                sidecar.write([row[index] if index < len(row) else "" for index in extra_indices])


def _detect_txt_delimiter(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig") as source:
        first_line = source.readline().strip()
    if not first_line:
        raise ValueError(f"{path}: file is empty or its first line is blank")

    counts = {delimiter: first_line.count(delimiter) for delimiter in ("\t", ";", "|")}
    counts = {delimiter: count for delimiter, count in counts.items() if count > 0}
    if not counts:
        raise ValueError(f"{path}: no supported delimiter found (tab, semicolon, or pipe)")
    return max(counts, key=counts.__getitem__)


def _parse_txt_line(line: str, delimiter: str) -> list[str]:
    """Parse one TXT line with single or double quotes and backslash escapes.

    :param line: Source line without its newline terminator.
    :param delimiter: Detected field delimiter.
    :return: Parsed fields.
    :raises ValueError: If a quoted field is not closed.
    """
    if not line:
        return []

    fields: list[str] = []
    current = ""
    quote: str | None = None
    index = 0
    while index < len(line):
        character = line[index]
        if character == "\\" and index + 1 < len(line):
            escaped = line[index + 1]
            replacements = {"n": "\n", "t": "\t", "r": "\r"}
            if escaped in ('"', "'", "\\", "n", "t", "r"):
                current += replacements.get(escaped, escaped)
                index += 2
                continue
        if character in ('"', "'") and quote is None:
            quote = character
            index += 1
            continue
        if character == quote:
            if index + 1 < len(line) and line[index + 1] == quote:
                current += character
                index += 2
                continue
            quote = None
            index += 1
            continue
        if character == delimiter and quote is None:
            fields.append(current)
            current = ""
            index += 1
            continue
        current += character
        index += 1

    if quote is not None:
        raise ValueError(f"Unclosed quote in line: {line[:50]}...")
    fields.append(current)
    return fields


def _txt_timestamps(
    path: Path,
    delimiter: str,
    timestamp_format: str | None,
    timestamp_column: str | None,
    date_column: str | None,
    time_column: str | None,
    timezone: _Timezone | None,
) -> Iterator[tuple[str, int, int]]:
    with open(path, "r", encoding="utf-8-sig") as source:
        header_line = source.readline().strip()
        raw_headers = _parse_txt_line(header_line, delimiter)
        headers = [header.strip().lower() for header in raw_headers]
        timestamp_indices = _find_timestamp_columns(
            headers, timestamp_column, date_column, time_column
        )
        for row_number, line in enumerate(source, start=2):
            stripped = line.strip()
            if not stripped:
                continue
            row = [field.strip() for field in _parse_txt_line(stripped, delimiter)]
            if len(row) != len(headers):
                raise ValueError(
                    f"{path}: row {row_number}: has {len(row)} columns, expected {len(headers)}"
                )
            try:
                text = _timestamp_text(row, *timestamp_indices)
                timestamp = _parse_timestamp(text, timestamp_format, timezone)
                wall_clock = _parse_timestamp(
                    text, timestamp_format, timezone, wall_clock=True
                )
            except (IndexError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{path}: row {row_number}: invalid timestamp: {error}") from error
            yield f"{path}: row {row_number}", timestamp, wall_clock


def infer_txt_period(
    path: str | Path,
    timestamp_format: str | None = None,
    timestamp_column: str | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    tz: str | None = None,
) -> str:
    """Infer the nominal v2 period from the smallest positive TXT timestamp delta.

    :param path: Source TXT path.
    :param timestamp_format: Explicit timestamp format, if needed.
    :param timestamp_column: Explicit combined timestamp column.
    :param date_column: Explicit date column for split timestamps.
    :param time_column: Explicit time column for split timestamps.
    :param tz: Timezone for naive timestamps.
    :return: Canonical TradingView period for the v2 header.
    :raises ValueError: If timestamps are malformed, unordered, or insufficient.
    """
    source_path = Path(path)
    delimiter = _detect_txt_delimiter(source_path)
    timezone = _parse_timezone_param(tz)
    return _infer_period(
        _txt_timestamps(
            source_path,
            delimiter,
            timestamp_format,
            timestamp_column,
            date_column,
            time_column,
            timezone,
        )
    )


def load_from_txt(
    writer: OHLCVWriter,
    path: str | Path,
    timestamp_format: str | None = None,
    timestamp_column: str | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    tz: str | None = None,
    extra_csv_path: str | Path | None = None,
) -> None:
    """Load delimited TXT rows into an open v2 writer without gap filling.

    Sidecar alignment follows the same accepted-record rule as :func:`load_from_csv`.
    Blank source lines are ignored and therefore do not create sidecar rows.

    :param writer: Open v2 writer.
    :param path: Source TXT path.
    :param timestamp_format: Explicit timestamp format, if needed.
    :param timestamp_column: Explicit combined timestamp column.
    :param date_column: Explicit date column for split timestamps.
    :param time_column: Explicit time column for split timestamps.
    :param tz: Timezone for naive timestamps.
    :param extra_csv_path: Optional destination for user extra columns.
    :raises ValueError: If headers, quoting, or any source row are malformed.
    """
    source_path = Path(path)
    delimiter = _detect_txt_delimiter(source_path)
    timezone = _parse_timezone_param(tz)

    with open(source_path, "r", encoding="utf-8-sig") as source:
        header_line = source.readline().strip()
        raw_headers = _parse_txt_line(header_line, delimiter)
        headers = [header.strip().lower() for header in raw_headers]
        timestamp_indices = _find_timestamp_columns(
            headers, timestamp_column, date_column, time_column
        )
        ohlcv_indices = _find_ohlcv_columns(headers)
        extra_indices, extra_headers = _extra_columns(raw_headers, headers)

        with _ExtraSidecar(extra_csv_path, extra_headers) as sidecar:
            for row_number, line in enumerate(source, start=2):
                stripped = line.strip()
                if not stripped:
                    continue
                row = [field.strip() for field in _parse_txt_line(stripped, delimiter)]
                if len(row) != len(headers):
                    raise ValueError(
                        f"{source_path}: row {row_number}: has {len(row)} columns, "
                        f"expected {len(headers)}"
                    )
                timestamp_text = _timestamp_text(row, *timestamp_indices)
                _parse_and_write_ohlcv_row(
                    writer,
                    source_path,
                    row_number,
                    timestamp_text,
                    row,
                    ohlcv_indices,
                    timestamp_format,
                    timezone,
                )
                sidecar.write([row[index] for index in extra_indices])


def _json_records(path: Path) -> list[Mapping[str, object]]:
    with open(path, "r", encoding="utf-8") as source:
        data = json.load(source)

    if isinstance(data, Mapping):
        for key in _JSON_WRAPPER_NAMES:
            candidate = data.get(key)
            if isinstance(candidate, list):
                data = candidate
                break
        else:
            raise ValueError(f"{path}: could not find an OHLCV data array in JSON")
    if not isinstance(data, list):
        raise ValueError(f"{path}: JSON must contain an array of OHLCV records")

    records: list[Mapping[str, object]] = []
    for index, record in enumerate(data, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}: record {index}: expected an object")
        records.append(record)
    return records


def _json_field_map(
    records: list[Mapping[str, object]],
    timestamp_field: str | None,
    date_field: str | None,
    time_field: str | None,
    mapping: dict[str, str] | None,
) -> dict[str, str | None]:
    if (date_field is None) != (time_field is None):
        raise ValueError("date_field and time_field must be supplied together")

    supplied = mapping or {}
    field_map: dict[str, str | None] = {
        "timestamp": supplied.get("timestamp", timestamp_field),
        "open": supplied.get("open", "open"),
        "high": supplied.get("high", "high"),
        "low": supplied.get("low", "low"),
        "close": supplied.get("close", "close"),
        "volume": supplied.get("volume", "volume"),
    }
    if field_map["timestamp"] is None and date_field is None:
        if not records:
            raise ValueError("Cannot find a timestamp field in an empty JSON array")
        first = records[0]
        for name in ("timestamp", "time", "date", "t"):
            if name in first:
                field_map["timestamp"] = name
                break
        if field_map["timestamp"] is None:
            raise ValueError("Could not find timestamp field")
    return field_map


def _json_text(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"JSON field {field_name!r} cannot be null")
    return str(cast(str | int | float | bool, value))


def _json_timestamp_text(
    record: Mapping[str, object],
    field_map: dict[str, str | None],
    date_field: str | None,
    time_field: str | None,
) -> str:
    if date_field is not None and time_field is not None:
        date_value = _json_text(record[date_field], date_field)
        time_value = _json_text(record[time_field], time_field)
        return f"{date_value} {time_value}"
    timestamp_name = field_map["timestamp"]
    assert timestamp_name is not None
    return _json_text(record[timestamp_name], timestamp_name)


def infer_json_period(
    path: str | Path,
    timestamp_format: str | None = None,
    timestamp_field: str | None = None,
    date_field: str | None = None,
    time_field: str | None = None,
    tz: str | None = None,
    mapping: dict[str, str] | None = None,
) -> str:
    """Infer the nominal v2 period from the smallest positive JSON timestamp delta.

    :param path: Source JSON path.
    :param timestamp_format: Explicit timestamp format, if needed.
    :param timestamp_field: Explicit combined timestamp field.
    :param date_field: Explicit date field for split timestamps.
    :param time_field: Explicit time field for split timestamps.
    :param tz: Timezone for naive timestamps.
    :param mapping: Optional logical-to-source field mapping.
    :return: Canonical TradingView period for the v2 header.
    :raises ValueError: If timestamps are malformed, unordered, or insufficient.
    """
    source_path = Path(path)
    records = _json_records(source_path)
    field_map = _json_field_map(
        records, timestamp_field, date_field, time_field, mapping
    )
    timezone = _parse_timezone_param(tz)

    def timestamps() -> Iterator[tuple[str, int, int]]:
        for index, record in enumerate(records, start=1):
            location = f"{source_path}: record {index}"
            try:
                text = _json_timestamp_text(record, field_map, date_field, time_field)
                timestamp = _parse_timestamp(text, timestamp_format, timezone)
                wall_clock = _parse_timestamp(
                    text, timestamp_format, timezone, wall_clock=True
                )
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{location}: invalid timestamp: {error}") from error
            yield location, timestamp, wall_clock

    return _infer_period(timestamps())


def load_from_json(
    writer: OHLCVWriter,
    path: str | Path,
    timestamp_format: str | None = None,
    timestamp_field: str | None = None,
    date_field: str | None = None,
    time_field: str | None = None,
    tz: str | None = None,
    mapping: dict[str, str] | None = None,
) -> None:
    """Load JSON records into an open v2 writer without filling timestamp gaps.

    :param writer: Open v2 writer.
    :param path: Source JSON path.
    :param timestamp_format: Explicit timestamp format, if needed.
    :param timestamp_field: Explicit combined timestamp field.
    :param date_field: Explicit date field for split timestamps.
    :param time_field: Explicit time field for split timestamps.
    :param tz: Timezone for naive timestamps.
    :param mapping: Optional logical-to-source field mapping.
    :raises ValueError: If the JSON structure or any record is malformed.
    """
    source_path = Path(path)
    records = _json_records(source_path)
    field_map = _json_field_map(
        records, timestamp_field, date_field, time_field, mapping
    )
    timezone = _parse_timezone_param(tz)

    for index, record in enumerate(records, start=1):
        location = f"{source_path}: record {index}"
        try:
            timestamp_text = _json_timestamp_text(record, field_map, date_field, time_field)
            timestamp = _parse_timestamp(timestamp_text, timestamp_format, timezone)
            open_name = field_map["open"]
            high_name = field_map["high"]
            low_name = field_map["low"]
            close_name = field_map["close"]
            volume_name = field_map["volume"]
            assert open_name is not None
            assert high_name is not None
            assert low_name is not None
            assert close_name is not None
            assert volume_name is not None
            candle = OHLCV(
                timestamp,
                float(cast(_JsonNumberInput, record[open_name])),
                float(cast(_JsonNumberInput, record[high_name])),
                float(cast(_JsonNumberInput, record[low_name])),
                float(cast(_JsonNumberInput, record[close_name])),
                float(cast(_JsonNumberInput, record[volume_name])),
            )
        except KeyError as error:
            raise ValueError(f"{location}: missing field {error.args[0]!r}") from error
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{location}: invalid OHLCV data: {error}") from error

        try:
            writer.write(candle)
        except ValueError as error:
            raise ValueError(f"{location}: {error}") from error
