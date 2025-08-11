"""
Fast and efficient OHLCV data reader/writer

The file is a binary file with the following 24 bytes structure:
 - timestamp: uint32 (4 bytes) - good until 2106 (I will fix this then, I promise ;))
 - open:     float32 (4 bytes)
 - high:     float32 (4 bytes)
 - low:      float32 (4 bytes)
 - close:    float32 (4 bytes)
 - volume:   float32 (4 bytes)

The .ohlcv format cannot have gaps in it. All gaps are filled with the previous close price and -1 volume.
"""

from typing import Iterator
import os
import mmap
import struct
from pathlib import Path
from io import BufferedWriter, BufferedRandom
from datetime import datetime, UTC

from pynecore.types.ohlcv import OHLCV

RECORD_SIZE = 24  # 6 * 4
STRUCT_FORMAT = 'Ifffff'  # I: uint32, f: float32

__all__ = ['OHLCVWriter', 'OHLCVReader']


def format_float(value: float) -> str:
    """Format float with max 8 decimal places, removing trailing zeros"""
    return f"{value:.8g}"


class OHLCVWriter:
    """
    Binary OHLCV data writer using direct file operations
    """

    __slots__ = ('path', '_file', '_size', '_start_timestamp', '_interval', '_current_pos', '_last_timestamp')

    def __init__(self, path: str | Path):
        self.path: str = str(path)
        self._file: BufferedWriter | BufferedRandom | None = None
        self._size: int = 0
        self._start_timestamp: int | None = None
        self._interval: int | None = None
        self._current_pos: int = 0
        self._last_timestamp: int | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_open(self) -> bool:
        """
        Check if file is open
        """
        return self._file is not None

    @property
    def size(self) -> int:
        """
        Number of records in the file
        """
        return self._size

    @property
    def start_timestamp(self) -> int | None:
        """
        Timestamp of the first record
        """
        return self._start_timestamp

    @property
    def start_datetime(self) -> datetime:
        """
        Datetime of the first record
        """
        return datetime.fromtimestamp(self._start_timestamp, UTC)

    @property
    def end_timestamp(self) -> int | None:
        """
        Timestamp of the last record
        """
        if self._start_timestamp is None or self._interval is None:
            return None
        return self._start_timestamp + self._interval * (self._size - 1)

    @property
    def end_datetime(self) -> datetime | None:
        """
        Datetime of the last record
        """
        if self.end_timestamp is None:
            return None
        return datetime.fromtimestamp(self.end_timestamp, UTC)

    @property
    def interval(self) -> int | None:
        """
        Interval between records
        """
        return self._interval

    def open(self) -> 'OHLCVWriter':
        """
        Open file for writing
        """
        # Open in rb+ mode to allow both reading and writing
        self._file = open(self.path, 'rb+') if os.path.exists(self.path) else open(self.path, 'wb+')
        self._size = os.path.getsize(self.path) // RECORD_SIZE

        # Read initial metadata if file exists
        if self._size >= 2:
            self._file.seek(0)
            first_timestamp = struct.unpack('I', self._file.read(4))[0]
            self._file.seek(RECORD_SIZE)
            second_timestamp = struct.unpack('I', self._file.read(4))[0]
            self._start_timestamp = first_timestamp
            self._interval = second_timestamp - first_timestamp
            assert self._interval is not None
            self._last_timestamp = first_timestamp + self._interval * (self._size - 1)

        # Position at end for appending
        self._file.seek(0, os.SEEK_END)
        self._current_pos = self._size

        return self

    def write(self, candle: OHLCV) -> None:
        """
        Write a single OHLCV candle at current position.
        If there is a gap between current and previous timestamp,
        fills it with the previous close price and -1 volume to indicate gap filling.

        :param candle: OHLCV data to write
        """
        if self._file is None:
            raise IOError("File not opened!")

        if self._size == 0:
            self._start_timestamp = candle.timestamp
        elif self._size == 1:
            # First interval detection
            assert self._start_timestamp is not None
            self._interval = candle.timestamp - self._start_timestamp
            if self._interval <= 0:
                raise ValueError(f"Invalid interval: {self._interval}")
        elif self._size >= 2:  # Changed from elif self._size == 2: to properly handle all cases
            # For the second candle, validate interval
            if self._size == 2:
                assert self._last_timestamp is not None and self._interval is not None
                current_interval = candle.timestamp - self._last_timestamp
                if current_interval > self._interval * 2:
                    # Truncate and restart
                    self.truncate()
                    self._start_timestamp = candle.timestamp
                    self._interval = None
                    self._last_timestamp = None
                    self._current_pos = 0
                    self._size = 0

            # Check chronological order
            if self._last_timestamp is not None and candle.timestamp <= self._last_timestamp:
                raise ValueError(
                    f"Timestamps must be in chronological order. Got {candle.timestamp} after {self._last_timestamp}")

            # Calculate expected timestamp and fill gaps
            if self._interval is not None and self._last_timestamp is not None:
                expected_ts = self._last_timestamp + self._interval

                # Fill gap if needed
                if candle.timestamp > expected_ts:
                    # Get previous candle's close price
                    self._file.seek((self._current_pos - 1) * RECORD_SIZE)
                    prev_data = struct.unpack(STRUCT_FORMAT, self._file.read(RECORD_SIZE))
                    prev_close = prev_data[4]  # 4th index is close price

                    # Fill gap with previous close and -1 volume (gap indicator)
                    while expected_ts < candle.timestamp:
                        gap_data = struct.pack(STRUCT_FORMAT,
                                               expected_ts, prev_close, prev_close,
                                               prev_close, prev_close, -1.0)
                        self._file.seek(self._current_pos * RECORD_SIZE)
                        self._file.write(gap_data)
                        self._current_pos += 1
                        self._size = max(self._size, self._current_pos)
                        expected_ts += self._interval

        # Write actual data
        self._file.seek(self._current_pos * RECORD_SIZE)
        data = struct.pack(STRUCT_FORMAT,
                           candle.timestamp, candle.open, candle.high,
                           candle.low, candle.close, candle.volume)
        self._file.write(data)
        self._file.flush()

        self._last_timestamp = candle.timestamp
        self._current_pos += 1
        self._size = max(self._size, self._current_pos)

    def seek_to_timestamp(self, timestamp: int) -> None:
        """
        Move write position to specific timestamp.
        Uses interval between bars to calculate position.
        """
        if self._interval is None or self._start_timestamp is None:
            return

        if timestamp < self._start_timestamp:
            raise ValueError("Timestamp before start of data")

        record_num = (timestamp - self._start_timestamp) // self._interval
        self.seek(int(record_num))

    def seek(self, position: int) -> None:
        """
        Move write position to specific record number
        """
        if position < 0:
            raise ValueError("Negative position not allowed")
        assert self._file is not None

        self._current_pos = position
        self._file.seek(position * RECORD_SIZE)

    def truncate(self) -> None:
        """
        Truncate file at current position.
        All data after current position will be deleted.
        """
        if self._file is None:
            raise IOError("File not opened!")

        # Calculate new size in bytes
        new_size = self._current_pos * RECORD_SIZE

        # Truncate the file
        self._file.truncate(new_size)
        self._size = self._current_pos

        # Update interval if we deleted too much
        if self._size < 2:
            self._interval = None
            if self._size == 0:
                self._start_timestamp = None

    def close(self):
        """
        Close the file
        """
        if self._file:
            self._file.close()
            self._file = None

    def load_from_csv(self, path: str | Path,
                      timestamp_format: str | None = None,
                      timestamp_column: str | None = None,
                      date_column: str | None = None,
                      time_column: str | None = None,
                      tz: str | None = None) -> None:
        """
        Load OHLCV data from CSV file using only builtin modules.

        :param path: Path to CSV file
        :param timestamp_format: Optional datetime fmt for parsing
        :param timestamp_column: Column name for timestamp (default tries: timestamp, time, date)
        :param date_column: When timestamp is split into date+time columns, date column name
        :param time_column: When timestamp is split into date+time columns, time column name
        :param tz: Timezone name (e.g. 'UTC', 'Europe/London', '+0100') for timestamp conversion
        """
        import csv
        from zoneinfo import ZoneInfo

        # Parse timezone
        timezone = None
        if tz:
            if tz.startswith(('+', '-')):
                # Handle UTC offset fmt (e.g. +0100, -0500)
                sign = 1 if tz.startswith('+') else -1
                hours = int(tz[1:3])
                minutes = int(tz[3:]) if len(tz) > 3 else 0
                from datetime import timezone as dt_timezone, timedelta
                timezone = dt_timezone(sign * timedelta(hours=hours, minutes=minutes))
            else:
                # Handle named timezone (e.g. UTC, Europe/London)
                try:
                    timezone = ZoneInfo(tz)
                except Exception as e:
                    raise ValueError(f"Invalid timezone {tz}: {e}")

        # Read CSV headers first
        with open(path, 'r') as f:
            reader = csv.reader(f)
            headers = [h.lower() for h in next(reader)]  # Case insensitive

            # Find timestamp column
            timestamp_idx = None
            date_idx = None
            time_idx = None

            if date_column and time_column:
                try:
                    date_idx = headers.index(date_column.lower())
                    time_idx = headers.index(time_column.lower())
                except ValueError:
                    raise ValueError(f"Date/time columns not found: {date_column}/{time_column}")
            else:
                timestamp_col = timestamp_column.lower() if timestamp_column else None
                if timestamp_col:
                    try:
                        timestamp_idx = headers.index(timestamp_col)
                    except ValueError:
                        raise ValueError(f"Timestamp column not found: {timestamp_col}")
                else:
                    # Try common names
                    for col in ['timestamp', 'time', 'date']:
                        try:
                            timestamp_idx = headers.index(col)
                            break
                        except ValueError:
                            continue

                    if timestamp_idx is None:
                        raise ValueError("Timestamp column not found!")

            # Find OHLCV columns
            try:
                o_idx = headers.index('open')
                h_idx = headers.index('high')
                l_idx = headers.index('low')
                c_idx = headers.index('close')
                v_idx = headers.index('volume')
            except ValueError as e:
                raise ValueError(f"Missing required column: {str(e)}")

            # Process data rows
            for row in reader:
                # Handle timestamp
                if date_idx is not None and time_idx is not None:
                    # Combine date and time
                    ts_str = f"{row[date_idx]} {row[time_idx]}"
                else:
                    ts_str = row[timestamp_idx]  # type: ignore

                # Convert timestamp
                try:
                    if ts_str.isdigit():
                        timestamp = int(ts_str)
                    else:
                        if timestamp_format:
                            dt = datetime.strptime(ts_str, timestamp_format)
                        else:
                            # Try common formats
                            for fmt in [
                                '%Y-%m-%d %H:%M:%S%z',  # 2024-01-08 19:00:00+0000
                                '%Y-%m-%d %H:%M:%S%Z',  # 2024-01-08 19:00:00UTC
                                '%Y-%m-%dT%H:%M:%S%z',  # 2024-01-08T19:00:00+0000
                                '%Y-%m-%d %H:%M:%S',
                                '%Y/%m/%d %H:%M:%S',
                                '%d.%m.%Y %H:%M:%S',
                                '%Y-%m-%dT%H:%M:%S',
                                '%Y-%m-%d %H:%M',
                                '%Y%m%d %H:%M:%S'
                            ]:
                                try:
                                    dt = datetime.strptime(ts_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                raise ValueError(f"Could not parse timestamp: {ts_str}")

                        # Set timezone if specified and convert to timestamp
                        if timezone:
                            dt = dt.replace(tzinfo=timezone)
                        timestamp = int(dt.timestamp())
                except Exception as e:
                    raise ValueError(f"Failed to parse timestamp '{ts_str}': {e}")

                # Write OHLCV data
                try:
                    self.write(OHLCV(
                        timestamp,
                        float(row[o_idx]),
                        float(row[h_idx]),
                        float(row[l_idx]),
                        float(row[c_idx]),
                        float(row[v_idx])
                    ))
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid data in row: {e}")

    def load_from_json(self, path: str | Path,
                       timestamp_format: str | None = None,
                       timestamp_field: str | None = None,
                       date_field: str | None = None,
                       time_field: str | None = None,
                       tz: str | None = None,
                       mapping: dict[str, str] | None = None) -> None:
        """
        Load OHLCV data from JSON file using only builtin modules.

        :param path: Path to JSON file
        :param timestamp_format: Optional datetime format for parsing
        :param timestamp_field: Field name for timestamp (default tries: timestamp, time, date, t)
        :param date_field: When timestamp is split, date field name
        :param time_field: When timestamp is split, time field name
        :param tz: Timezone name (e.g. 'UTC', 'Europe/London', '+0100')
        :param mapping: Optional field mapping, e.g. {'timestamp': 't', 'volume': 'vol'}
        """
        import json
        from datetime import datetime
        from zoneinfo import ZoneInfo

        # Parse timezone
        timezone = None
        if tz:
            if tz.startswith(('+', '-')):
                # Handle UTC offset format
                sign = 1 if tz.startswith('+') else -1
                hours = int(tz[1:3])
                minutes = int(tz[3:]) if len(tz) > 3 else 0
                from datetime import timezone as dt_timezone, timedelta
                timezone = dt_timezone(sign * timedelta(hours=hours, minutes=minutes))
            else:
                # Handle named timezone
                try:
                    timezone = ZoneInfo(tz)
                except Exception as e:
                    raise ValueError(f"Invalid timezone {tz}: {e}")

        # Setup field mapping
        mapping = mapping or {}
        field_map = {
            'timestamp': mapping.get('timestamp', timestamp_field),
            'open': mapping.get('open', 'open'),
            'high': mapping.get('high', 'high'),
            'low': mapping.get('low', 'low'),
            'close': mapping.get('close', 'close'),
            'volume': mapping.get('volume', 'volume')
        }

        # Load JSON file
        data = None
        with open(path, 'r') as f:
            data = json.load(f)

        # Ensure we have a list of records
        if isinstance(data, dict):
            # Some APIs wrap the data in an object
            for key in ['data', 'candles', 'ohlcv', 'results']:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                raise ValueError("Could not find OHLCV data array in JSON")

        if not isinstance(data, list):
            raise ValueError("JSON must contain an array of OHLCV records")

        # Find timestamp field if not specified
        if not field_map['timestamp'] and not (date_field and time_field):
            common_names = ['timestamp', 'time', 'date', 't']
            for record in data[:1]:  # Check just first record
                for name in common_names:
                    if name in record:
                        field_map['timestamp'] = name
                        break
                if field_map['timestamp']:
                    break
            if not field_map['timestamp']:
                raise ValueError("Could not find timestamp field")

        # Process records
        for record in data:
            # Get timestamp
            try:
                if date_field and time_field:
                    # Combine date and time
                    ts_str = f"{record[date_field]} {record[time_field]}"
                else:
                    ts_str = str(record[field_map['timestamp']])

                # Convert timestamp
                if ts_str.isdigit():
                    # Handle millisecond timestamps
                    ts = int(ts_str)
                    if ts > 253402300799:  # 9999-12-31 23:59:59
                        ts //= 1000
                    timestamp = ts
                else:
                    dt = None
                    # Parse datetime string
                    if timestamp_format:
                        dt = datetime.strptime(ts_str, timestamp_format)
                    else:
                        # Try common formats
                        for fmt in [
                            '%Y-%m-%d %H:%M:%S%z',  # 2024-01-08 19:00:00+0000
                            '%Y-%m-%d %H:%M:%S%Z',  # 2024-01-08 19:00:00UTC
                            '%Y-%m-%dT%H:%M:%S%z',  # 2024-01-08T19:00:00+0000
                            '%Y-%m-%d %H:%M:%S',
                            '%Y/%m/%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%dT%H:%M:%SZ',
                            '%Y-%m-%d %H:%M',
                            '%Y%m%d %H:%M:%S'
                        ]:
                            try:
                                dt = datetime.strptime(ts_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            raise ValueError(f"Could not parse timestamp: {ts_str}")

                    # Set timezone and convert to timestamp
                    if timezone:
                        dt = dt.replace(tzinfo=timezone)
                    timestamp = int(dt.timestamp())

                # Get OHLCV values
                try:
                    self.write(OHLCV(
                        timestamp,
                        float(record[field_map['open']]),
                        float(record[field_map['high']]),
                        float(record[field_map['low']]),
                        float(record[field_map['close']]),
                        float(record[field_map['volume']])
                    ))
                except KeyError as e:
                    raise ValueError(f"Missing field in record: {e}")
                except ValueError as e:
                    raise ValueError(f"Invalid value in record: {e}")

            except Exception as e:
                raise ValueError(f"Failed to process record: {e}")


class OHLCVReader:
    """
    Very fast OHLCV data reader using memory mapping.
    """

    __slots__ = ('path', '_file', '_mmap', '_size', '_start_timestamp', '_interval')

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._file = None
        self._mmap = None
        self._size = 0
        self._start_timestamp = None
        self._interval = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def size(self) -> int:
        """
        Number of records in the file
        """
        return self._size

    @property
    def start_timestamp(self) -> int | None:
        """
        Timestamp of the first record
        """
        return self._start_timestamp

    @property
    def start_datetime(self) -> datetime:
        """
        Datetime of the first record
        """
        return datetime.fromtimestamp(self._start_timestamp, UTC)

    @property
    def end_timestamp(self) -> int | None:
        """
        Timestamp of the last record
        """
        return self._start_timestamp + self._interval * (self._size - 1) if self._interval else None

    @property
    def end_datetime(self) -> datetime:
        """
        Datetime of the last record
        """
        return datetime.fromtimestamp(self.end_timestamp, UTC)

    @property
    def interval(self) -> int | None:
        """
        Interval between records
        """
        return self._interval

    def open(self) -> 'OHLCVReader':
        """
        Open file and create memory mapping
        """
        self._file = open(self.path, 'rb')
        if os.path.getsize(self.path) > 0:
            # Detect if this is a text file masquerading as binary OHLCV
            self._file.seek(0)
            first_chunk = self._file.read(32)
            self._file.seek(0)  # Reset position

            try:
                # If 256 bytes decode as ASCII, it's definitely not binary OHLCV
                first_chunk.decode('ascii')

                # If we get here, it's text - show error with CLI fix
                raise ValueError(
                    f"Text file detected with .ohlcv extension!\n"
                    f"To convert CSV to binary OHLCV format:\n"
                    f"  pyne data convert-from {Path(self.path).with_suffix('.csv')} "
                    f"--symbol YOUR_SYMBOL --provider custom"
                )
            except UnicodeDecodeError:
                # Can't decode as ASCII → it's binary, proceed normally
                pass

            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._size = os.path.getsize(self.path) // RECORD_SIZE

            if self._size >= 2:
                self._start_timestamp = struct.unpack('I', self._mmap[0:4])[0]
                second_timestamp = struct.unpack('I', self._mmap[RECORD_SIZE:RECORD_SIZE + 4])[0]
                self._interval = second_timestamp - self._start_timestamp

        return self

    def __iter__(self) -> Iterator[OHLCV]:
        """
        Iterate through all candles
        """
        for pos in range(self._size):
            yield self.read(pos)

    def read(self, position: int) -> OHLCV:
        """
        Read a single candle at given position
        """
        if position < 0 or position >= self._size:
            raise IndexError("Position out of range")

        assert self._mmap is not None

        offset = position * RECORD_SIZE
        data = struct.unpack(STRUCT_FORMAT, self._mmap[offset:offset + RECORD_SIZE])
        return OHLCV(*data, extra_fields={})

    def read_from(self, start_timestamp: int, end_timestamp: int | None = None, skip_gaps: bool = True) \
            -> Iterator[OHLCV]:
        """
        Read bars starting from timestamp, using direct position calculation.

        :param start_timestamp: Start timestamp
        :param end_timestamp: End timestamp, if None, read until the end
        :param skip_gaps: Skip gaps in data, the writer fill gaps with the last value with -1 volume,
                          this will skip them (default)
        :raises ValueError: If start_timestamp is after the last bar
        """
        if not self._size or not self._interval:
            return

        # Calculate start and end positions
        start_pos, end_pos = self.get_positions(start_timestamp, end_timestamp)

        # Yield the calculated range
        for pos in range(start_pos, end_pos):
            ohlcv = self.read(pos)
            # Skip gaps if needed
            if skip_gaps and ohlcv.volume < 0:
                continue
            yield ohlcv

    def close(self):
        """
        Close file and memory mapping
        """
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None

    def get_positions(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> tuple[int, int]:
        """
        Get start and end positions for given timestamps

        :param start_timestamp: Start timestamp
        :param end_timestamp: End timestamp
        :return: Tuple of start and end positions
        """
        if not self._size or not self._interval:
            return 0, 0
        assert self._start_timestamp is not None

        # Calculate start position
        if start_timestamp is None:
            start_pos = 0
        else:
            start_diff = start_timestamp - self._start_timestamp
            if start_diff < 0:
                start_pos = 0
            else:
                start_pos = min(start_diff // self._interval, self._size - 1)

        # Calculate end position if provided
        if end_timestamp is None:
            end_pos = self._size
        else:
            end_diff = end_timestamp - self._start_timestamp
            end_pos = min(end_diff // self._interval + 1, self._size)

        return start_pos, end_pos

    def get_size(self, start_timestamp: int | None = None, end_timestamp: int | None = None) -> int:
        """
        Get number of records between timestamps

        :param start_timestamp: Start timestamp
        :param end_timestamp: End timestamp
        :return: Number of records
        """
        if not self._size or not self._interval:
            return 0

        start_pos, end_pos = self.get_positions(start_timestamp, end_timestamp)
        return end_pos - start_pos

    def save_to_csv(self, path: str, as_datetime=False) -> None:
        """
        Save OHLCV data to CSV file

        :param path: Path to the CSV file
        :param as_datetime: Save timestamp as datetime string
        """

        with open(path, 'w') as f:
            if as_datetime:
                f.write('time,open,high,low,close,volume\n')
            else:
                f.write('timestamp,open,high,low,close,volume\n')
            for candle in self:
                if as_datetime:
                    f.write(f"{datetime.fromtimestamp(candle.timestamp, UTC)},{format_float(candle.open)},"
                            f"{format_float(candle.high)},{format_float(candle.low)},{format_float(candle.close)},"
                            f"{format_float(candle.volume)}\n")
                else:
                    f.write(f"{candle.timestamp},{format_float(candle.open)},{format_float(candle.high)},"
                            f"{format_float(candle.low)},{format_float(candle.close)},{format_float(candle.volume)}\n")

    def save_to_json(self, path: str, as_datetime: bool = False) -> None:
        """
        Save OHLCV data to JSON file.

        The output fmt is either:
        [
            {
                "timestamp": 1234567890,  // or "time": "2024-01-07 12:34:56+00:00" if as_datetime is True
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0
            },
            ...
        ]

        :param path: Path to save the JSON file
        :param as_datetime: If True, convert timestamps to ISO fmt datetime strings
        """
        import json

        data = []
        for candle in self:
            if as_datetime:
                item = {
                    "time": datetime.fromtimestamp(candle.timestamp, UTC).isoformat(),
                    "open": format_float(candle.open),
                    "high": format_float(candle.high),
                    "low": format_float(candle.low),
                    "close": format_float(candle.close),
                    "volume": format_float(candle.volume)
                }
            else:
                item = {
                    "timestamp": candle.timestamp,
                    "open": format_float(candle.open),
                    "high": format_float(candle.high),
                    "low": format_float(candle.low),
                    "close": format_float(candle.close),
                    "volume": format_float(candle.volume)
                }
            data.append(item)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)  # Use indent for human-readable fmt  # noqa
