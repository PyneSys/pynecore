"""
Read-only support for the legacy headerless OHLCV format.

Each record uses the following 24-byte native-endian structure:

- timestamp: uint32 seconds
- open: float32
- high: float32
- low: float32
- close: float32
- volume: float32

The reader exposes timestamps in milliseconds so its public values use the same
unit as the current OHLCV format. Legacy phantom records have negative volume.
"""
from typing import Iterator

import csv
import json
import mmap
import os
import struct
from datetime import UTC, datetime
from pathlib import Path

from pynecore.types.ohlcv import OHLCV

RECORD_SIZE = 24
STRUCT_FORMAT = "Ifffff"

__all__ = ["OHLCVReader"]


def _format_float(value: float) -> str:
    """Format a float with at most eight significant digits."""
    return f"{value:.8g}"


class OHLCVReader:
    """
    Read legacy OHLCV data using memory mapping.

    Raw legacy records store timestamps in seconds. All timestamps exposed by
    this reader, including :class:`OHLCV` values, properties, and range
    parameters, are Unix timestamps in milliseconds.
    """

    __slots__ = (
        "path",
        "_file",
        "_mmap",
        "_size",
        "_start_timestamp",
        "_interval",
        "_extra_data",
        "_extra_headers",
    )

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._file = None
        self._mmap = None
        self._size = 0
        self._start_timestamp = None
        self._interval = None
        self._extra_data: list[dict[str, int | float | str]] | None = None
        self._extra_headers: list[str] | None = None

    def __enter__(self) -> "OHLCVReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def size(self) -> int:
        """Return the number of records in the file."""
        return self._size

    @property
    def start_timestamp(self) -> int | None:
        """Return the first record timestamp in milliseconds."""
        return self._start_timestamp

    @property
    def start_datetime(self) -> datetime:
        """Return the first record datetime."""
        assert self._start_timestamp is not None
        return datetime.fromtimestamp(self._start_timestamp / 1000, UTC)

    @property
    def end_timestamp(self) -> int | None:
        """Return the last record timestamp in milliseconds."""
        if self._size == 0:
            return None

        if self._mmap and self._size > 0:
            last_record_offset = (self._size - 1) * RECORD_SIZE
            seconds = struct.unpack("I", self._mmap[last_record_offset:last_record_offset + 4])[0]
            return seconds * 1000

        return None

    @property
    def end_datetime(self) -> datetime:
        """Return the last record datetime."""
        timestamp = self.end_timestamp
        assert timestamp is not None
        return datetime.fromtimestamp(timestamp / 1000, UTC)

    @property
    def interval(self) -> int | None:
        """Return the inferred interval in milliseconds."""
        return self._interval

    def open(self) -> "OHLCVReader":
        """Open the file and create its memory mapping."""
        self._file = open(self.path, "rb")
        size = os.path.getsize(self.path)
        if size > 0:
            if size % RECORD_SIZE != 0:
                self._file.seek(0)
                first_chunk = self._file.read(256)
                self._file.seek(0)

                try:
                    first_chunk.decode("ascii")
                except UnicodeDecodeError:
                    pass
                else:
                    raise ValueError(
                        "Text file detected with .ohlcv extension!\n"
                        "To convert CSV to binary OHLCV format:\n"
                        f"  pyne data convert-from {Path(self.path).with_suffix('.csv')} "
                        "--symbol YOUR_SYMBOL --provider custom"
                    )

            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._size = size // RECORD_SIZE
            first_seconds = struct.unpack("I", self._mmap[0:4])[0]
            self._start_timestamp = first_seconds * 1000

            if self._size >= 2:
                second_seconds = struct.unpack("I", self._mmap[RECORD_SIZE:RECORD_SIZE + 4])[0]
                self._interval = (second_seconds - first_seconds) * 1000

        self._load_extra_csv()
        return self

    def _load_extra_csv(self) -> None:
        """
        Load position-aligned extra fields from the sidecar CSV file.
        """
        extra_path = Path(self.path).with_suffix(".extra.csv")
        if not extra_path.exists():
            return

        with open(extra_path, "r", newline="") as extra_file:
            reader = csv.reader(extra_file)
            headers = next(reader, None)
            if not headers:
                return

            self._extra_headers = headers
            rows_raw: list[list[str]] = []
            col_is_numeric: list[bool | None] = [None] * len(headers)

            for row in reader:
                rows_raw.append(row)
                for index, value in enumerate(row):
                    if col_is_numeric[index] is None and value and value.lower() not in ("", "nan", "na"):
                        try:
                            float(value)
                            col_is_numeric[index] = True
                        except ValueError:
                            col_is_numeric[index] = False

            numeric_columns = [value if value is not None else False for value in col_is_numeric]
            extra_data: list[dict[str, int | float | str]] = []
            for row in rows_raw:
                parsed: dict[str, int | float | str] = {}
                for index, header in enumerate(headers):
                    value = row[index] if index < len(row) else ""
                    if numeric_columns[index]:
                        if not value or value.lower() in ("nan", "na"):
                            parsed[header] = float("nan")
                        else:
                            parsed[header] = float(value)
                    else:
                        parsed[header] = value
                extra_data.append(parsed)
            self._extra_data = extra_data

    def __iter__(self) -> Iterator[OHLCV]:
        """Iterate over all records."""
        for position in range(self._size):
            yield self.read(position)

    def read(self, position: int) -> OHLCV:
        """
        Read one record.

        :param position: Zero-based record position.
        :return: The record with its timestamp in milliseconds.
        :raises IndexError: If the position is outside the file.
        """
        if position < 0 or position >= self._size:
            raise IndexError("Position out of range")

        assert self._mmap is not None
        offset = position * RECORD_SIZE
        data = struct.unpack(STRUCT_FORMAT, self._mmap[offset:offset + RECORD_SIZE])

        extra = {}
        if self._extra_data is not None and position < len(self._extra_data):
            extra = self._extra_data[position]

        return OHLCV(data[0] * 1000, *data[1:], extra_fields=extra)

    def read_from(
        self,
        start_timestamp: int,
        end_timestamp: int | None = None,
        skip_gaps: bool = True,
    ) -> Iterator[OHLCV]:
        """
        Read records starting from a timestamp using direct position calculation.

        :param start_timestamp: Start timestamp in milliseconds.
        :param end_timestamp: End timestamp in milliseconds, or ``None`` to read to the end.
        :param skip_gaps: Skip legacy phantom records with negative volume.
        :return: An iterator over matching records.
        """
        if not self._size:
            return

        start_pos, end_pos = self.get_positions(start_timestamp, end_timestamp)
        for position in range(start_pos, end_pos):
            ohlcv = self.read(position)
            if skip_gaps and ohlcv.volume < 0:
                continue
            yield ohlcv

    def close(self) -> None:
        """Close the file and memory mapping."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
        self._extra_data = None
        self._extra_headers = None

    def _timestamp_at(self, position: int) -> int:
        """
        Return the timestamp of one record in milliseconds.

        :param position: Zero-based record position.
        :return: The record timestamp in milliseconds.
        """
        assert self._mmap is not None
        return struct.unpack_from("I", self._mmap, position * RECORD_SIZE)[0] * 1000

    def _search(self, timestamp: int, after_equal: bool) -> int:
        """
        Binary-search the stored timestamps for an insertion position.

        :param timestamp: Timestamp in milliseconds to locate.
        :param after_equal: Place the position after an exact match instead of before it.
        :return: Insertion position in ``0..size``.
        """
        low = 0
        high = self._size
        while low < high:
            middle = (low + high) // 2
            value = self._timestamp_at(middle)
            if value < timestamp or (after_equal and value == timestamp):
                low = middle + 1
            else:
                high = middle
        return low

    def get_positions(
        self,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ) -> tuple[int, int]:
        """
        Return positions for a timestamp range.

        The bounds are located by searching the stored timestamps rather than by
        projecting them from the first interval: legacy files may hold a single
        record, duplicate timestamps, or irregular spacing, and interval arithmetic
        then lands on the wrong record — replaying stale bars or dropping matching
        ones. The records are ordered, so the search is exact for every file.

        :param start_timestamp: Start timestamp in milliseconds.
        :param end_timestamp: End timestamp in milliseconds.
        :return: The half-open start and end positions.
        """
        if not self._size:
            return 0, 0

        start_pos = 0 if start_timestamp is None else self._search(start_timestamp, False)
        end_pos = self._size if end_timestamp is None else self._search(end_timestamp, True)
        if end_pos < start_pos:
            return start_pos, start_pos
        return start_pos, end_pos

    def get_size(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> int:
        """
        Return the number of records in a timestamp range.

        :param start_timestamp: Start timestamp in milliseconds.
        :param end_timestamp: End timestamp in milliseconds.
        :return: The number of matching records.
        """
        if not self._size:
            return 0

        start_pos, end_pos = self.get_positions(start_timestamp, end_timestamp)
        return end_pos - start_pos

    def save_to_csv(self, path: str, as_datetime: bool = False) -> None:
        """
        Save the records to a CSV file.

        :param path: Destination path.
        :param as_datetime: Write timestamps as datetime strings.
        """
        with open(path, "w") as output_file:
            if as_datetime:
                output_file.write("time,open,high,low,close,volume\n")
            else:
                output_file.write("timestamp,open,high,low,close,volume\n")
            for candle in self:
                if candle.volume == -1:
                    continue
                if as_datetime:
                    output_file.write(
                        f"{datetime.fromtimestamp(candle.timestamp / 1000, UTC)},"
                        f"{_format_float(candle.open)},{_format_float(candle.high)},"
                        f"{_format_float(candle.low)},{_format_float(candle.close)},"
                        f"{_format_float(candle.volume)}\n"
                    )
                else:
                    output_file.write(
                        f"{candle.timestamp},{_format_float(candle.open)},{_format_float(candle.high)},"
                        f"{_format_float(candle.low)},{_format_float(candle.close)},"
                        f"{_format_float(candle.volume)}\n"
                    )

    def save_to_json(self, path: str, as_datetime: bool = False) -> None:
        """
        Save the records to a JSON file.

        :param path: Destination path.
        :param as_datetime: Write timestamps as ISO datetime strings.
        """
        data = []
        for candle in self:
            if candle.volume == -1:
                continue
            if as_datetime:
                item = {
                    "time": datetime.fromtimestamp(candle.timestamp / 1000, UTC).isoformat(),
                    "open": _format_float(candle.open),
                    "high": _format_float(candle.high),
                    "low": _format_float(candle.low),
                    "close": _format_float(candle.close),
                    "volume": _format_float(candle.volume),
                }
            else:
                item = {
                    "timestamp": candle.timestamp,
                    "open": _format_float(candle.open),
                    "high": _format_float(candle.high),
                    "low": _format_float(candle.low),
                    "close": _format_float(candle.close),
                    "volume": _format_float(candle.volume),
                }
            data.append(item)

        with open(path, "w") as output_file:
            json.dump(data, output_file, indent=2)
