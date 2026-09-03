from __future__ import annotations
from typing import Iterator, Optional, Literal
import io
import mmap
import csv
import queue
import threading
from pathlib import Path
from datetime import datetime, UTC
from math import copysign as _copysign

from pynecore.types.ohlcv import OHLCV
from pynecore.types.na import NA, na_float

DO_NOTHING = -1
WRITE_TUPLE = 0
WRITE_DICT = 1
WRITE_OHLCV = 2
STOP = 3
FLUSH = 4


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
                 '_worker', '_error', '_is_open', '_stopping', '_lock',
                 '_idle_time', '_dialect')

    def __init__(self, path: Path, *,
                 buffer_size: int = 32768,
                 queue_size: int = 4096,
                 float_fmt: str = '',
                 timestamp_as_iso: bool = True,
                 idle_time: float = 0.016,
                 dialect: Literal['lf', 'excel', 'excel-tab', 'unix'] = 'lf',
                 headers: tuple | list | None = None):
        """
        :param path: Output file path
        :param buffer_size: Internal buffer size in bytes
        :param queue_size: Size of the command queue
        :param float_fmt: Format string for float values. The default writes the
                          shortest string that reads back as the same double, so
                          the file loses nothing — the same choice TradingView's
                          own export makes. A fixed digit count (``.8g``) would
                          quantize every value on the way out.
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
        # True once close() has queued the STOP: the worker is on its way out,
        # so nothing queued after it can still reach the file
        self._stopping = False
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

        # With the default (empty) float_fmt, ``repr(x)`` emits the very same
        # shortest-roundtrip digits as ``'{:}'.format(x)`` for an exact float, but
        # skips the whole __format__ dispatch. Verified byte-identical on 2 000 000
        # random doubles plus 0.0, -0.0, ±inf, 5e-324, 1e308, 1e23, 1/3, 2^53 and the
        # 1e-3/1e-4/1e16/1e17 notation switch points. The guard has to be an exact
        # type test: a float *subclass* renders its type name (repr(np.float64(1.5))
        # -> 'np.float64(1.5)'), so those keep going through fmt.
        fast_repr = not self._float_fmt

        def fmt_float(x: object) -> str:
            # Canonicalize na to Pine's "NaN". A native nan would format lowercase
            # ("nan") via fmt; an NA object formats as "NaN" already but we keep the
            # branch representation-agnostic so both map to the same token.
            if not (x == x):
                return "NaN"
            if fast_repr and type(x) is float:
                # An integral value prints as the integer, the way TradingView's
                # export writes it ("14", "2451"): a Pine int is a double at
                # runtime, so a bar index or a count arrives here as a float, and
                # the float-typed Volume column shows TradingView does the same
                # for every integral value. Beyond 2**53 a double has no
                # fractional digits to drop, so the shortest repr stands.
                if x.is_integer() and -9007199254740992.0 < x < 9007199254740992.0 and (
                        x != 0.0 or _copysign(1.0, x) > 0.0):
                    return repr(int(x))
                return repr(x)
            return fmt.format(x)

        row = []

        assert self._file is not None
        # Bound locally: close() drops the attribute once it stops waiting for
        # us, and a worker still draining must not trip over a None there
        file = self._file

        def drain() -> None:
            """Move the in-memory buffer into the file."""
            file.write(buffer.getvalue())
            file.flush()
            buffer.truncate(0)
            buffer.seek(0)

        try:
            while True:
                try:
                    cmd, data = self._queue.get(timeout=self._idle_time)
                except queue.Empty:
                    # Idle: nothing is coming, so put whatever is buffered into
                    # the file — a half-full threshold here would leave a small
                    # tail sitting in memory until close()
                    if buffer.tell() > 0:
                        drain()
                    continue

                stop = False
                try:
                    if cmd == STOP:
                        stop = True

                    elif cmd == FLUSH:
                        # The waiter is released only AFTER the file write, so
                        # flush() never returns while rows are still sitting in
                        # the in-memory buffer. A failing write must release it
                        # too, with the error recorded FIRST so the waiter
                        # reports the failure instead of returning cleanly.
                        try:
                            if buffer.tell() > 0:
                                drain()
                        except Exception as e:
                            self._error = e
                            raise
                        finally:
                            if data is not None:
                                data.set()

                    else:
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
                            # OHLCV timestamps are Unix milliseconds.
                            if self._timestamp_as_iso:
                                row.append(
                                    datetime.fromtimestamp(data.timestamp / 1000, UTC).isoformat())
                            else:
                                row.append(str(data.timestamp))

                            # Format OHLCV values
                            ohlcv = (data.open, data.high, data.low, data.close, data.volume)
                            row.extend(fmt_float(x) for x in ohlcv)
                            # Format extra fields
                            if data.extra_fields:
                                for value in data.extra_fields.values():
                                    if isinstance(value, float):
                                        row.append(fmt_float(value))
                                    elif isinstance(value, NA):
                                        row.append("NaN")
                                    else:
                                        row.append(str(value))

                        # Tuple or dict data
                        else:
                            for value in data:
                                if isinstance(value, float):
                                    row.append(fmt_float(value))
                                elif isinstance(value, datetime):
                                    if self._timestamp_as_iso:
                                        row.append(value.isoformat())
                                    else:
                                        # What ``str(datetime)`` yields: ISO with a space
                                        row.append(value.isoformat(sep=' '))
                                elif isinstance(value, NA):
                                    row.append("NaN")
                                else:
                                    row.append(str(value))

                        # Write row to buffer
                        writer.writerow(row)

                        # Write if buffer is half full
                        if buffer.tell() >= self._buffer_size // 2:
                            drain()
                finally:
                    # Every successful get() must be balanced, error paths and
                    # STOP included — an unfinished item blocks queue.join()
                    self._queue.task_done()

                if stop:
                    break

        except Exception as e:
            self._error = e
        finally:
            # Final flush. A failure here means queued rows never reached the
            # file, so it must be recorded — close() reports it instead of
            # returning as if everything had been written. An error already
            # recorded is the earlier (root) one and wins.
            if buffer.tell() > 0:
                try:
                    drain()
                except Exception as e:
                    if self._error is None:
                        self._error = e
            # A dead worker must not turn a pending flush() or close() into a
            # permanent block: release everything still queued
            self._abandon_queue()

    def _abandon_queue(self) -> None:
        """Discard everything left in the queue, releasing any flush waiter.

        Called from the worker's exit path. Without it a producer blocked in
        ``flush()`` — or in a bounded ``put()`` against a full queue — would
        wait for a thread that is never coming back.
        """
        while True:
            try:
                cmd, data = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd == FLUSH and data is not None:
                    data.set()
            finally:
                self._queue.task_done()

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
        :raises RuntimeError: If the writer is shutting down, or the writer thread
                              has died with an error
        """
        if not self._is_open:
            raise RuntimeError("Writer not opened!")
        # A record queued behind the STOP is dropped by the worker's exit path,
        # so accepting it would hand back an acknowledgement that never holds
        if self._stopping:
            raise RuntimeError("Writer is closing!")
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
        :raises RuntimeError: If the writer is shutting down, or the writer thread
                              has died with an error
        """
        if not self._is_open:
            raise RuntimeError("Writer not opened!")
        if self._stopping:  # queued behind the STOP, see write_dict()
            raise RuntimeError("Writer is closing!")
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
        :raises RuntimeError: If the writer is shutting down, or the writer thread
                              has died with an error
        """
        if not self._is_open:
            raise RuntimeError("Writer not opened!")
        if self._stopping:  # queued behind the STOP, see write_dict()
            raise RuntimeError("Writer is closing!")
        if self._error:
            raise RuntimeError(f"Writer thread error: {self._error}!")

        try:
            self._queue.put((WRITE_OHLCV, candle), timeout=timeout)
            return True
        except queue.Full:
            return False

    def flush(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until every record queued so far has reached the file.

        The queue is FIFO, so the acknowledgement of this command implies every
        earlier record was formatted, buffered AND written out.

        :param timeout: Optional timeout in seconds for queuing the command
        :return: True if the flush completed, False if the command could not be queued
        :raises RuntimeError: If writer thread has died with an error
        """
        # Same lock as close(): a command queued for a worker that is already
        # on its way out would wait for an acknowledgement nobody sends
        with self._lock:
            # A worker on its way out releases the FLUSH waiter without writing
            # anything, so the acknowledgement would be meaningless
            if not self._is_open or self._stopping:
                return False
            if self._error:
                raise RuntimeError(f"Writer thread error: {self._error}!")

            done = threading.Event()
            try:
                self._queue.put((FLUSH, done), timeout=timeout)
            except queue.Full:
                return False
            done.wait()
            if self._error:
                raise RuntimeError(f"Writer thread error: {self._error}!")
            return True

    def close(self, timeout: Optional[float] = None):
        """
        Close the CSV file and stop the worker thread.

        :param timeout: Optional timeout in seconds to wait for remaining writes
        :raises TimeoutError: If the worker did not stop in time; the file stays open so
                              the call can be retried, but once the STOP is queued the
                              writer no longer accepts records
        :raises RuntimeError: If writer thread has died with an error
        """
        with self._lock:
            if not self._is_open:
                return

            # A worker that never received the STOP — or that is still draining
            # when the timeout expires — keeps writing to the file. Closing it
            # here would destroy records write() already accepted, so nothing is
            # torn down until the thread is confirmed gone.
            worker = self._worker
            if worker is not None and worker.is_alive():
                # A retry after a timed-out join must not queue a second STOP:
                # the first one is still on its way and the extra command would
                # only be dropped by the worker's exit path
                if not self._stopping:
                    try:
                        self._queue.put((STOP, None), timeout=timeout)
                    except queue.Full:
                        raise TimeoutError(
                            f"Could not signal the writer thread of {self.path} to stop!") from None
                    self._stopping = True

                worker.join(timeout=timeout)
                if worker.is_alive():
                    raise TimeoutError(f"Writer thread of {self.path} did not stop in time!")

            self._worker = None

            # Close the file
            if self._file:
                self._file.close()
                self._file = None

            self._is_open = False
            self._stopping = False

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
                if value == "NaN" or value == "na" or value == "nan":
                    extra[name] = na_float
                else:
                    try:
                        # A number is a double, the way every Pine number is
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

            # Parse timestamp into Unix milliseconds, the OHLCV timestamp unit
            time_field = row[fi['time']]
            if time_field.isdigit():
                # Numeric exports carry either seconds or milliseconds; the decimal
                # width separates them (10 digits of seconds reach the year 2286,
                # while 10 digits of milliseconds would still be in 1970)
                timestamp = int(time_field)
                if len(time_field) <= 10:
                    timestamp *= 1000
            else:
                try:
                    dt = datetime.fromisoformat(time_field).astimezone(UTC)
                    timestamp = round(dt.timestamp() * 1000)
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
