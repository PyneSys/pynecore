from __future__ import annotations
from typing import Iterator, Optional, Literal
import io
import mmap
import csv
import queue
import threading
from pathlib import Path
from datetime import datetime, UTC

from pynecore.types.ohlcv import OHLCV
from pynecore.types.na import NA

DO_NOTHING = -1
WRITE_TUPLE = 0
WRITE_DICT = 1
WRITE_OHLCV = 2
STOP = 3


class DialectLF(csv.excel):
    """CSV dialect with line feed as newline character"""
    lineterminator = '\n'


csv.register_dialect("lf", DialectLF)


class CSVWriter:
    """
    Fast CSV writer for OHLCV data with extra fields.
    Uses a background thread and buffering for better performance.
    """
    __slots__ = ('path', '_file', '_buffer_size', '_float_fmt',
                 '_timestamp_as_iso', '_headers', '_queue',
                 '_worker', '_error', '_is_open', '_lock',
                 '_idle_time', '_dialect')

    def __init__(self, path: Path, *,
                 buffer_size: int = 32768,
                 queue_size: int = 4096,
                 float_fmt: str = '.8g',
                 timestamp_as_iso: bool = True,
                 idle_time: float = 0.016,
                 dialect: Literal['lf', 'excel', 'excel-tab', 'unix'] = 'lf',
                 headers: tuple | list | None = None):
        """
        :param path: Output file path
        :param buffer_size: Internal buffer size in bytes
        :param queue_size: Size of the command queue
        :param float_fmt: Format string for float values
        :param timestamp_as_iso: If True, timestamps will be written as ISO datetime strings
        :param idle_time: Idle time in seconds before flushing the buffer
        :param dialect: CSV dialect, one of 'excel', 'excel-tab', 'unix'
        :param headers: Optional list of headers to write
        """
        self.path = path
        self._idle_time = idle_time
        self._dialect = dialect

        self._file: io.TextIOWrapper | None = None
        self._buffer_size = buffer_size
        self._float_fmt = float_fmt
        self._timestamp_as_iso = timestamp_as_iso
        self._headers = headers

        # Thread-safe queue for commands
        self._queue = queue.Queue(maxsize=queue_size)
        self._worker = None
        self._error = None
        self._is_open = False
        self._lock = threading.Lock()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_open(self) -> bool:
        """Check if the writer is open"""
        return self._is_open

    def _worker_thread(self):
        """Background worker thread for handling I/O operations"""
        buffer = io.StringIO()
        writer = csv.writer(buffer, dialect=self._dialect)

        # Format string for float values
        fmt = '{:' + self._float_fmt + '}'
        row = []

        assert self._file is not None

        try:
            while True:
                try:
                    cmd, data = self._queue.get(timeout=self._idle_time)
                except queue.Empty:
                    # Write buffer if idle
                    if buffer.tell() > self._buffer_size // 2:
                        self._file.write(buffer.getvalue())
                        self._file.flush()
                        buffer.truncate(0)
                        buffer.seek(0)
                    continue

                if cmd == STOP:
                    break

                # Write header if needed
                if not self._headers:
                    if cmd == WRITE_DICT:
                        headers = list(data.keys())
                        writer.writerow(headers)
                    elif cmd == WRITE_OHLCV:
                        headers = ['time', 'open', 'high', 'low', 'close', 'volume']
                        if data.extra_fields:
                            headers.extend(data.extra_fields.keys())
                        writer.writerow(headers)
                    else:
                        raise ValueError(f"No headers provided!")
                    self._headers = headers

                # Format Timestamp
                row.clear()

                # Raw dictionary data
                if cmd == WRITE_DICT:
                    data = data.values()

                # OHLCV data
                if cmd == WRITE_OHLCV:
                    if self._timestamp_as_iso:
                        row.append(datetime.fromtimestamp(data.timestamp, UTC).isoformat())
                    else:
                        row.append(str(data.timestamp))

                    # Format OHLCV values
                    row.extend(fmt.format(x) for x in (data.open, data.high, data.low, data.close, data.volume))
                    # Format extra fields
                    if data.extra_fields:
                        for value in data.extra_fields.values():
                            if isinstance(value, float):
                                row.append(fmt.format(value))
                            else:
                                row.append(str(value))

                # Tuple or dict data
                else:
                    for value in data:
                        if isinstance(value, float):
                            row.append(fmt.format(value))
                        elif isinstance(value, datetime):
                            if self._timestamp_as_iso:
                                row.append(value.isoformat())
                            else:
                                row.append(str(value))
                        else:
                            row.append(str(value))

                # Write row to buffer
                writer.writerow(row)
                self._queue.task_done()

                # Write if buffer is half full
                if buffer.tell() >= self._buffer_size // 2:
                    self._file.write(buffer.getvalue())
                    self._file.flush()
                    buffer.truncate(0)
                    buffer.seek(0)

        except Exception as e:
            self._error = e
        finally:
            # Final flush
            if buffer.tell() > 0:
                try:
                    self._file.write(buffer.getvalue())
                    self._file.flush()
                except:  # noqa
                    pass

    def open(self) -> CSVWriter:
        """Open the CSV file and start the worker thread"""
        with self._lock:
            if self._is_open:
                return self

            # Open file for writing
            file = open(self.path, 'w', buffering=self._buffer_size)
            self._file = file
            self._is_open = True

            # Write headers if provided
            if self._headers:
                writer = csv.writer(file, dialect=self._dialect)
                writer.writerow(self._headers)

            # Start the worker thread
            self._worker = threading.Thread(target=self._worker_thread)
            self._worker.daemon = True  # Thread dies with the program
            self._worker.start()

            return self

    def write_dict(self, data: dict[str, int | float | str], timeout: Optional[float] = None) -> bool:
        """
        Write a raw dict record.

        :param data: The dict to write
        :param timeout: Optional timeout in seconds
        :return: True if write command was queued, False on timeout
        :raises RuntimeError: If writer thread has died with an error
        """
        if not self._is_open:
            raise RuntimeError("Writer not opened!")
        if self._error:
            raise RuntimeError(f"Writer thread error: {self._error}!")

        try:
            self._queue.put((WRITE_DICT, data), timeout=timeout)
            return True
        except queue.Full:
            return False

    def write(self, *data: object, timeout: Optional[float] = None) -> bool:
        """
        Write raw data

        :param data: the data to write
        :param timeout: Optional timeout in seconds
        :return: True if write command was queued, False on timeout
        :raises RuntimeError: If writer thread has died with an error
        """
        if not self._is_open:
            raise RuntimeError("Writer not opened!")
        if self._error:
            raise RuntimeError(f"Writer thread error: {self._error}!")

        try:
            self._queue.put((WRITE_TUPLE, data), timeout=timeout)
            return True
        except queue.Full:
            return False

    def write_ohlcv(self, candle: OHLCV, timeout: Optional[float] = None) -> bool:
        """
        Write a single OHLCV record.

        :param candle: The OHLCV record to write
        :param timeout: Optional timeout in seconds
        :return: True if write command was queued, False on timeout
        :raises RuntimeError: If writer thread has died with an error
        """
        if not self._is_open:
            raise RuntimeError("Writer not opened!")
        if self._error:
            raise RuntimeError(f"Writer thread error: {self._error}!")

        try:
            self._queue.put((WRITE_OHLCV, candle), timeout=timeout)
            return True
        except queue.Full:
            return False

    def flush(self, timeout: Optional[float] = None):
        """
        Wait for all pending writes to be processed by the worker thread.

        :param timeout: Optional timeout in seconds
        """
        if self._is_open and self._queue is not None:
            self._queue.join()

    def close(self, timeout: Optional[float] = None):
        """
        Close the CSV file and stop the worker thread.

        :param timeout: Optional timeout in seconds to wait for remaining writes
        """
        with self._lock:
            if not self._is_open:
                return

            # Signal the worker to stop
            try:
                self._queue.put((STOP, None), timeout=timeout)
            except queue.Full:
                pass  # We'll stop anyway

            # Wait for the worker to finish
            if self._worker:
                self._worker.join(timeout=timeout)
                self._worker = None

            # Close the file
            if self._file:
                self._file.close()
                self._file = None

            self._is_open = False

            # Re-raise any worker thread error
            if self._error:
                raise RuntimeError(f"Writer thread error: {self._error}")


class CSVReader:
    """
    Simple CSV reader for OHLCV data with support for extra fields.
    Sequential access only.
    """

    __slots__ = ('path', '_file', '_mmap', '_headers', '_dialect', '_has_headers',
                 '_field_indices', '_extra_fields', '_is_valid_ohlcv')

    def __init__(self, path: Path):
        self.path: Path = path
        self._file: io.BufferedReader | None = None
        self._headers: list[str] | None = None
        self._dialect: type[csv.Dialect] | None = None
        self._has_headers: bool = True
        self._field_indices: dict[str, int] | None = None
        self._extra_fields: dict[str, int] | None = None
        self._mmap: mmap.mmap | None = None
        self._is_valid_ohlcv: bool = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> CSVReader:
        """Open the CSV file"""
        # Open file in binary mode for memory mapping
        file = open(self.path, 'rb')
        self._file = file
        mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self._mmap = mm

        # Read first line to detect CSV format and headers
        first_line = mm.readline().decode('utf-8')

        # Detect dialect
        dialect = csv.Sniffer().sniff(first_line)  # type: ignore
        self._dialect = dialect

        # Check if we have headers
        self._has_headers = csv.Sniffer().has_header(first_line)

        _is_tv = False
        if self._has_headers:
            # Parse headers
            headers = next(csv.reader([first_line], dialect=dialect))
            _is_tv = (headers[0] == 'time' and headers[1] == 'open' and headers[2] == 'high'
                      and headers[3] == 'low' and headers[4] == 'close')
        else:
            # Default headers for standard OHLCV
            headers = ['time', 'open', 'high', 'low', 'close', 'volume']
        self._headers = headers

        # Reopen file to reset position
        file.seek(0)

        # Create case-insensitive header mapping
        header_map = {h.lower(): i for i, h in enumerate(headers) if not _is_tv or i < 6 or h == "Volume"}

        # Get field indices for OHLCV data with case-insensitive matching
        try:
            fi: dict[str, int] = {
                # support both 'time' and 'timestamp'
                'time': header_map['time'] if 'time' in header_map else header_map['timestamp'],
                # OHLCV fields
                'open': header_map['open'],
                'high': header_map['high'],
                'low': header_map['low'],
                'close': header_map['close'],
                'volume': header_map['volume']
            }
            self._is_valid_ohlcv = True
        except KeyError:
            fi = {
                'time': header_map.get('data/time', 0),
            }
        self._field_indices = fi

        # Get extra field indices
        self._extra_fields = {
            name: idx for idx, name in enumerate(headers)
            if idx not in fi.values()
        }

        return self

    def _parse_extra_fields(self, row: list[str]) -> dict:
        """Parse extra fields from a row"""
        extra = {}
        if self._extra_fields is None:
            return extra

        for name, idx in self._extra_fields.items():
            name = name.replace('&quot;', '"')  # Handle HTML quote entities
            try:
                value = row[idx]
                if value == "NaN" or value == "na":
                    extra[name] = NA()
                else:
                    try:
                        # Try converting the value to an integer
                        extra[name] = int(value)
                    except ValueError:
                        try:
                            # Fallback to converting the value to a float
                            extra[name] = float(value)
                        except ValueError:
                            # Value is not a valid numeric representation
                            extra[name] = value
            except (ValueError, IndexError):
                continue
        return extra

    def _read_records(self, target_pos: Optional[int] = None) -> Iterator[tuple[int, OHLCV]]:
        """
        Internal method to read records, optionally stopping at target_pos.
        Returns (position, candle) tuples.
        """
        if not self._file:
            raise RuntimeError("File not opened!")
        assert self._mmap is not None
        assert self._field_indices is not None
        assert self._dialect is not None
        fi = self._field_indices
        dialect = self._dialect

        # Reset position
        self._mmap.seek(0)

        # Create a text IO wrapper for the mmap object
        text_io = io.TextIOWrapper(io.BytesIO(self._mmap))
        reader = csv.reader(text_io, dialect=dialect)

        # Skip header if needed
        if self._has_headers:
            next(reader)

        for pos, row in enumerate(reader):
            # Stop if we reached target position
            if target_pos is not None and pos > target_pos:
                break

            if not row:  # Skip empty rows
                continue

            # Parse timestamp
            time_field = row[fi['time']]
            if time_field.isdigit():
                timestamp = int(time_field)
            else:
                try:
                    dt = datetime.fromisoformat(time_field).astimezone(UTC)
                    timestamp = int(dt.timestamp())
                except ValueError:
                    raise ValueError(f"Invalid time format: {time_field}")

            # Create OHLCV object
            try:
                if self._is_valid_ohlcv:
                    candle = OHLCV(
                        timestamp=timestamp,
                        open=float(row[fi['open']]),
                        high=float(row[fi['high']]),
                        low=float(row[fi['low']]),
                        close=float(row[fi['close']]),
                        volume=float(row[fi['volume']]),
                        extra_fields=self._parse_extra_fields(row)
                    )
                else:
                    candle = OHLCV(
                        timestamp=timestamp,
                        open=float('nan'),
                        high=float('nan'),
                        low=float('nan'),
                        close=float('nan'),
                        volume=float('nan'),
                        extra_fields=self._parse_extra_fields(row)
                    )

            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid data in row {pos + 1}: {e}")

            yield pos, candle

    def read(self, position: int) -> OHLCV:
        """
        Read a single candle at given position.
        Must read sequentially from the start to reach the position.
        """
        if position < 0:
            raise IndexError("Position cannot be negative")

        for pos, candle in self._read_records(position):
            if pos == position:
                return candle

        raise IndexError("Position out of range")

    def read_from(self, start_timestamp: int, end_timestamp: int | None = None) -> Iterator[OHLCV]:
        """
        Read bars starting from timestamp.
        Must read sequentially until finding matching timestamps.
        """
        for _, candle in self._read_records():
            if candle.timestamp >= start_timestamp:
                if end_timestamp is None or candle.timestamp <= end_timestamp:
                    yield candle
                else:
                    break

    def __iter__(self) -> Iterator[OHLCV]:
        """Iterate through all candles"""
        for _, candle in self._read_records():
            yield candle

    def close(self):
        """Close file and memory mapping"""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
