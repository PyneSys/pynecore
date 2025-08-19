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
from typing import Iterator, cast

import csv
import json
import math
import mmap
import os
import struct
from collections import Counter
from collections.abc import Buffer
from datetime import datetime, time, timedelta, timezone as dt_timezone, UTC
from io import BufferedWriter, BufferedRandom
from math import gcd as math_gcd
from pathlib import Path
from zoneinfo import ZoneInfo

from pynecore.types.ohlcv import OHLCV
from ..core.syminfo import SymInfoInterval

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

    __slots__ = ('path', '_file', '_size', '_start_timestamp', '_interval', '_current_pos', '_last_timestamp',
                 '_price_changes', '_price_decimals', '_last_close', '_analyzed_tick_size',
                 '_analyzed_price_scale', '_analyzed_min_move', '_confidence',
                 '_trading_hours', '_analyzed_opening_hours', '_truncate')

    def __init__(self, path: str | Path, truncate: bool = False):
        self.path: str = str(path)
        self._file: BufferedWriter | BufferedRandom | None = None
        self._truncate: bool = truncate
        self._size: int = 0
        self._start_timestamp: int | None = None
        self._interval: int | None = None
        self._current_pos: int = 0
        self._last_timestamp: int | None = None
        # Tick size analysis
        self._price_changes: list[float] = []
        self._price_decimals: set[int] = set()
        self._last_close: float | None = None
        self._analyzed_tick_size: float | None = None
        self._analyzed_price_scale: int | None = None
        self._analyzed_min_move: int | None = None
        self._confidence: float = 0.0
        # Trading hours analysis
        self._trading_hours: dict[tuple[int, int], int] = {}  # (weekday, hour) -> count
        self._analyzed_opening_hours: list | None = None

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

    @property
    def analyzed_tick_size(self) -> float | None:
        """
        Automatically detected tick size from price data
        """
        if self._analyzed_tick_size is None and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._analyzed_tick_size

    @property
    def analyzed_price_scale(self) -> int | None:
        """
        Automatically detected price scale from price data
        """
        if self._analyzed_price_scale is None and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._analyzed_price_scale

    @property
    def analyzed_min_move(self) -> int | None:
        """
        Automatically detected min move (usually 1)
        """
        if self._analyzed_min_move is None and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._analyzed_min_move

    @property
    def tick_analysis_confidence(self) -> float:
        """
        Confidence of tick size analysis (0.0 to 1.0)
        """
        if self._confidence == 0.0 and len(self._price_changes) >= 10:
            self._analyze_tick_size()
        return self._confidence

    @property
    def analyzed_opening_hours(self) -> list | None:
        """
        Automatically detected opening hours from trading activity
        Returns list of SymInfoInterval tuples or None if not enough data
        """
        if self._analyzed_opening_hours is None and self._has_enough_data_for_opening_hours():
            self._analyze_opening_hours()
        return self._analyzed_opening_hours

    def open(self) -> 'OHLCVWriter':
        """
        Open file for writing
        """
        # If truncate is True, always open in write mode to clear existing data
        if self._truncate:
            self._file = open(self.path, 'wb+')
        else:
            # Open in rb+ mode to allow both reading and writing
            self._file = open(self.path, 'rb+') if os.path.exists(self.path) else open(self.path, 'wb+')
        self._size = os.path.getsize(self.path) // RECORD_SIZE

        # Read initial metadata if file exists
        if self._size >= 2:
            self._file.seek(0)
            data: Buffer = self._file.read(4)
            first_timestamp = struct.unpack('I', data)[0]
            self._file.seek(RECORD_SIZE)
            data: Buffer = self._file.read(4)
            second_timestamp = struct.unpack('I', data)[0]
            self._start_timestamp = first_timestamp
            self._interval = second_timestamp - first_timestamp
            assert self._interval is not None
            self._last_timestamp = first_timestamp + self._interval * (self._size - 1)

        # Position at end for appending
        self._file.seek(0, os.SEEK_END)
        self._current_pos = self._size

        # Collect trading hours from existing data for analysis
        if self._size > 0 and not self._truncate:
            self._collect_existing_trading_hours()

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
            # Check chronological order
            if self._last_timestamp is not None and candle.timestamp <= self._last_timestamp:
                raise ValueError(
                    f"Timestamps must be in chronological order. Got {candle.timestamp} after {self._last_timestamp}")

            # Check if we found a smaller interval (indicates initial interval was wrong due to gap)
            if self._interval is not None and self._last_timestamp is not None:
                current_interval = candle.timestamp - self._last_timestamp

                # If we find a smaller interval, the initial one was wrong (had a gap)
                if 0 < current_interval < self._interval:
                    # Rebuild file with correct interval
                    self._rebuild_with_correct_interval(current_interval)
                    # Now write the current candle with the corrected setup
                    self.write(candle)
                    return

            # Calculate expected timestamp and fill gaps
            if self._interval is not None and self._last_timestamp is not None:
                expected_ts = self._last_timestamp + self._interval

                # Fill gap if needed
                if candle.timestamp > expected_ts:
                    # Get previous candle's close price
                    self._file.seek((self._current_pos - 1) * RECORD_SIZE)
                    data: Buffer = self._file.read(RECORD_SIZE)
                    prev_data = struct.unpack(STRUCT_FORMAT, data)
                    prev_close = prev_data[4]  # 4th index is close price

                    # Fill gap with previous close and -1 volume (gap indicator)
                    while expected_ts < candle.timestamp:
                        gap_data: Buffer = struct.pack(STRUCT_FORMAT,
                                                       expected_ts, prev_close, prev_close,
                                                       prev_close, prev_close, -1.0)
                        self._file.seek(self._current_pos * RECORD_SIZE)
                        self._file.write(gap_data)
                        self._current_pos += 1
                        self._size = max(self._size, self._current_pos)
                        expected_ts += self._interval

        # Write actual data
        self._file.seek(self._current_pos * RECORD_SIZE)
        data: Buffer = struct.pack(STRUCT_FORMAT,
                                   candle.timestamp, candle.open, candle.high,
                                   candle.low, candle.close, candle.volume)
        self._file.write(data)
        self._file.flush()

        # Collect data for tick size analysis
        self._collect_price_data(candle)

        # Collect trading hours data
        self._collect_trading_hours(candle)

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

    def _collect_price_data(self, candle: OHLCV) -> None:
        """
        Collect price data for tick size analysis during writing.
        """
        # Collect price changes
        if self._last_close is not None:
            change = abs(candle.close - self._last_close)
            if change > 0 and len(self._price_changes) < 1000:  # Limit to 1000 samples
                self._price_changes.append(change)

        # Collect decimal places
        for price in [candle.open, candle.high, candle.low, candle.close]:
            if price != int(price):  # Has decimal component
                price_str = f"{price:.15f}".rstrip('0').rstrip('.')
                if '.' in price_str:
                    decimals = len(price_str.split('.')[1])
                    self._price_decimals.add(decimals)

        self._last_close = candle.close

    def _analyze_tick_size(self) -> None:
        """
        Analyze collected price data to determine tick size using multiple methods.
        """
        if not self._price_changes:
            # No data, use defaults
            self._analyzed_tick_size = 0.01
            self._analyzed_price_scale = 100
            self._analyzed_min_move = 1
            self._confidence = 0.1
            return

        # Try histogram-based method first for better noise handling
        histogram_tick = self._calculate_histogram_tick()

        if histogram_tick[0] > 0 and histogram_tick[1] > 0.7:
            # High confidence histogram result, use it directly
            self._analyzed_tick_size = histogram_tick[0]
            self._analyzed_price_scale = int(round(1.0 / histogram_tick[0]))
            self._analyzed_min_move = 1
            self._confidence = histogram_tick[1]
            return

        # Fall back to other methods
        # Method 1: Most frequent small change
        freq_tick = self._calculate_frequency_tick()

        # Method 2: Decimal places analysis
        decimal_tick = self._calculate_decimal_tick()

        # Combine methods with weighted confidence (no GCD)
        tick_size, confidence = self._combine_tick_estimates_no_gcd(freq_tick, decimal_tick)

        # Calculate price scale and min move
        if tick_size > 0:
            self._analyzed_tick_size = tick_size
            self._analyzed_price_scale = int(round(1.0 / tick_size))
            self._analyzed_min_move = 1
            self._confidence = confidence
        else:
            # Fallback to defaults
            self._analyzed_tick_size = 0.01
            self._analyzed_price_scale = 100
            self._analyzed_min_move = 1
            self._confidence = 0.1

    def _calculate_frequency_tick(self) -> tuple[float, float]:
        """
        Calculate tick size based on most frequent small changes.
        Returns (tick_size, confidence)
        """
        if len(self._price_changes) < 10:
            return 0, 0

        # Apply float32 filtering first
        filtered_changes = []
        for c in self._price_changes[:100]:
            if c > 0:
                # Convert to float32 and back
                float32_val = struct.unpack('f', cast(Buffer, struct.pack('f', c)))[0]
                # Round to reasonable precision for float32
                rounded = round(float32_val, 6)
                if rounded > 0:
                    filtered_changes.append(rounded)

        if len(filtered_changes) < 5:
            return 0, 0

        # Find most frequent changes
        counter = Counter(filtered_changes)
        most_common = counter.most_common(10)

        if not most_common:
            return 0, 0

        # Find GCD of frequent changes to get base tick
        frequent_changes = [change for change, count in most_common if count >= 2]
        if len(frequent_changes) >= 2:
            # Convert to integers for GCD
            scale = 1000000  # 6 decimal places
            int_changes = [int(round(c * scale)) for c in frequent_changes]

            # Calculate GCD
            result = int_changes[0]
            for val in int_changes[1:]:
                result = math_gcd(result, val)

            tick_size = result / scale

            # Confidence based on how many changes match this tick
            matches = sum(1 for c in filtered_changes
                          if abs(round(c / tick_size) * tick_size - c) < tick_size * 0.1)
            confidence = min(matches / len(filtered_changes), 1.0)
            return tick_size, confidence * 0.7  # Medium weight

        return 0, 0

    def _calculate_histogram_tick(self) -> tuple[float, float]:
        """
        Calculate tick size using histogram-based clustering approach.
        This method is robust to float32 noise.
        Returns (tick_size, confidence)
        """
        if len(self._price_changes) < 10:
            return 0, 0

        # Common tick sizes to test (from 1 to 0.00001)
        candidate_ticks = [
            1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001,
            0.0005, 0.0001, 0.00005, 0.00001, 0.000001
        ]

        best_tick = 0
        best_score = 0

        # Filter out zero changes and convert to float32 precision
        changes = []
        for change in self._price_changes[:200]:  # Use more samples for histogram
            if change > 0:
                # Round to float32 precision
                float32_val = struct.unpack('f', cast(Buffer, struct.pack('f', change)))[0]
                changes.append(float32_val)

        if len(changes) < 5:
            return 0, 0

        # Get min non-zero change to establish scale
        min_change = min(changes)
        avg_change = sum(changes) / len(changes)

        for tick in candidate_ticks:
            # Skip ticks that are too small (less than 1/10 of smallest change)
            if tick < min_change * 0.1:
                continue

            # Skip ticks that are way too large
            if tick > avg_change * 10:
                continue

            # Round all changes to this tick size
            rounded = [round(c / tick) * tick for c in changes]

            # Calculate how well the rounding fits
            errors = [abs(c - r) for c, r in zip(changes, rounded)]
            max_error = max(errors)

            # Key insight: if max error is less than tick/2, this tick captures the grid well
            if max_error < tick * 0.5:
                # Count how many changes are multiples of this tick (within tolerance)
                tolerance = tick * 0.1
                multiples = sum(1 for c in changes if abs(round(c / tick) * tick - c) < tolerance)
                multiple_ratio = multiples / len(changes)

                # Score based on how many values are clean multiples
                if multiple_ratio > 0.7:  # Most values are clean multiples
                    score = multiple_ratio

                    # Prefer larger ticks (less precision) when scores are similar
                    # This helps choose 0.00001 over 0.000001 when both fit
                    score *= (1.0 + tick * 100)  # Small bonus for larger ticks

                    if score > best_score:
                        best_score = score
                        best_tick = tick

        # If no good tick found with strict criteria, fall back to simple analysis
        if best_tick == 0:
            # Find the most common order of magnitude in changes
            magnitudes = []
            for c in changes:
                if c > 0:
                    # Find order of magnitude
                    mag = 10 ** math.floor(math.log10(c))
                    magnitudes.append(mag)

            if magnitudes:
                # Most common magnitude
                counter = Counter(magnitudes)
                common_mag = counter.most_common(1)[0][0]
                # Use tick as 1/10 of common magnitude
                best_tick = common_mag / 10
                best_score = 0.5

        # Calculate confidence based on score
        if best_score > 0.8:
            confidence = 0.9
        elif best_score > 0.6:
            confidence = 0.7
        else:
            confidence = best_score

        return best_tick, confidence

    def _calculate_decimal_tick(self) -> tuple[float, float]:
        """
        Calculate tick size based on decimal places.
        Returns (tick_size, confidence)
        """
        if not self._price_decimals:
            # No decimals found, probably integer prices
            return 1.0, 0.5

        # Filter out noise from float representation
        # If we have 15 decimals, it's likely float noise
        valid_decimals = [d for d in self._price_decimals if d <= 10]

        if not valid_decimals:
            # All decimals are noise, assume 2 decimal places (cents)
            return 0.01, 0.3

        # Use most common valid decimal places
        max_decimals = max(valid_decimals)
        tick_size = 10 ** (-max_decimals)

        # Lower confidence for decimal-only method
        return tick_size, 0.5

    @staticmethod
    def _combine_tick_estimates_no_gcd(freq: tuple[float, float],
                                       decimal: tuple[float, float]) -> tuple[float, float]:
        """
        Combine tick size estimates from frequency and decimal methods only.
        Returns (tick_size, confidence)
        """
        estimates = []

        if freq[0] > 0 and freq[1] > 0:
            estimates.append(freq)
        if decimal[0] > 0 and decimal[1] > 0:
            estimates.append(decimal)

        if not estimates:
            return 0.01, 0.1  # Default fallback

        # Use highest confidence estimate
        best = max(estimates, key=lambda x: x[1])
        return best

    def _collect_trading_hours(self, candle: OHLCV) -> None:
        """
        Collect trading hours data from timestamps.
        Only collect for candles with actual volume (not gaps).
        """
        if candle.volume <= 0:
            return  # Skip gaps

        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(candle.timestamp, tz=None)  # Local time

        # Get weekday (1=Monday, 7=Sunday) and hour
        weekday = dt.isoweekday()
        hour = dt.hour

        # Count occurrences
        key = (weekday, hour)
        self._trading_hours[key] = self._trading_hours.get(key, 0) + 1

    def _collect_existing_trading_hours(self) -> None:
        """
        Collect trading hours data from existing file for opening hours analysis.
        Only samples a subset of data for performance reasons.
        """
        if not self._file or self._size == 0:
            return

        # Save current position
        current_pos = self._file.tell()

        try:
            # Sample data: read every Nth record for performance
            # For large files, we don't need to read everything
            sample_interval = max(1, self._size // 1000)  # Sample up to 1000 points

            for i in range(0, self._size, sample_interval):
                self._file.seek(i * RECORD_SIZE)
                data = self._file.read(RECORD_SIZE)

                if len(data) == RECORD_SIZE:
                    # Unpack the record
                    timestamp, open_val, high, low, close, volume = \
                        struct.unpack('Ifffff', cast(Buffer, data))

                    # Only collect if volume > 0 (real trading)
                    if volume > 0:
                        dt = datetime.fromtimestamp(timestamp, tz=None)
                        weekday = dt.isoweekday()
                        hour = dt.hour
                        key = (weekday, hour)
                        self._trading_hours[key] = self._trading_hours.get(key, 0) + 1

        finally:
            # Restore file position
            self._file.seek(current_pos)

    def _has_enough_data_for_opening_hours(self) -> bool:
        """
        Check if we have enough data to analyze opening hours based on timeframe.
        """
        if not self._trading_hours or not self._interval:
            return False

        # For daily or larger timeframes
        if self._interval >= 86400:  # >= 1 day
            # We need at least a few days to see a pattern
            unique_days = len(set(day for day, hour in self._trading_hours.keys()))
            return unique_days >= 3  # At least 3 different days

        # For intraday timeframes
        # Check if we have at least some meaningful data
        # We need enough to see a pattern
        data_points = sum(self._trading_hours.values())
        points_per_hour = 3600 / self._interval
        hours_covered = data_points / points_per_hour

        # Need at least 2 hours of data to detect any pattern
        # This allows even short sessions to be analyzed
        return hours_covered >= 2

    def _analyze_opening_hours(self) -> None:
        """
        Analyze collected trading hours to determine opening hours pattern.
        Works for both intraday and daily timeframes.
        """
        if not self._trading_hours:
            self._analyzed_opening_hours = None
            return

        # For daily or larger timeframes, analyze which days have trading
        if self._interval and self._interval >= 86400:  # >= 1 day
            self._analyzed_opening_hours = []
            days_with_trading = set(day for day, hour in self._trading_hours.keys())

            # Check if it's 24/7 (all 7 days have trading)
            if len(days_with_trading) == 7:
                # 24/7 trading pattern
                for day in range(1, 8):
                    self._analyzed_opening_hours.append(SymInfoInterval(
                        day=day,
                        start=time(0, 0, 0),
                        end=time(23, 59, 59)
                    ))
            elif days_with_trading <= {1, 2, 3, 4, 5}:  # Monday-Friday only
                # Business days pattern (stock/forex)
                for day in range(1, 6):
                    self._analyzed_opening_hours.append(SymInfoInterval(
                        day=day,
                        start=time(9, 30, 0),  # Default to US market hours
                        end=time(16, 0, 0)
                    ))
            else:
                # Mixed pattern - include all days that have trading
                for day in sorted(days_with_trading):
                    self._analyzed_opening_hours.append(SymInfoInterval(
                        day=day,
                        start=time(0, 0, 0),  # Default to full day for daily data
                        end=time(23, 59, 59)
                    ))
            return

        # For intraday data, analyze hourly patterns
        # Check if it's 24/7 trading (crypto pattern)
        total_hours = len(self._trading_hours)
        if total_hours >= 168 * 0.7:  # 70% of all hours in a week (lowered threshold)
            # Check if all hours have similar activity
            counts = list(self._trading_hours.values())
            avg_count = sum(counts) / len(counts)
            variance = sum((c - avg_count) ** 2 for c in counts) / len(counts)

            # If low variance, it's likely 24/7
            if variance < avg_count * 0.5:
                self._analyzed_opening_hours = []
                for day in range(1, 8):
                    self._analyzed_opening_hours.append(SymInfoInterval(
                        day=day,
                        start=time(0, 0, 0),
                        end=time(23, 59, 59)
                    ))
                return

        # Analyze per-day patterns for intraday
        self._analyzed_opening_hours = []

        for day in range(1, 8):  # Monday to Sunday
            # Get all hours for this day
            day_hours = [(hour, count) for (d, hour), count in self._trading_hours.items() if d == day]

            if not day_hours:
                continue  # No trading on this day

            # Sort by hour
            day_hours.sort(key=lambda x: x[0])

            # Find continuous trading periods
            periods = []
            current_start = None
            current_end = None

            # Threshold: consider an hour active if it has at least 20% of average activity
            total_count = sum(count for _, count in day_hours)
            if total_count == 0:
                continue
            avg_hour_count = total_count / len(day_hours)
            threshold = avg_hour_count * 0.2

            for hour, count in day_hours:
                if count >= threshold:
                    if current_start is None:
                        current_start = hour
                        current_end = hour
                    else:
                        current_end = hour
                else:
                    if current_start is not None:
                        periods.append((current_start, current_end))
                        current_start = None
                        current_end = None

            # Add last period if exists
            if current_start is not None:
                periods.append((current_start, current_end))

            # Convert periods to SymInfoInterval
            for start_hour, end_hour in periods:
                self._analyzed_opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(start_hour, 0, 0),
                    end=time(end_hour, 59, 59)
                ))

        # If no opening hours detected, default to business hours
        if not self._analyzed_opening_hours:
            for day in range(1, 6):  # Monday to Friday
                self._analyzed_opening_hours.append(SymInfoInterval(
                    day=day,
                    start=time(9, 30, 0),
                    end=time(16, 0, 0)
                ))

    def _rebuild_with_correct_interval(self, new_interval: int) -> None:
        """
        Rebuild the entire file with the correct interval when a smaller interval is detected.
        This happens when initial interval was wrong due to gaps.

        :param new_interval: The correct interval to use
        """
        import tempfile
        import shutil

        if not self._file or self._size == 0:
            return

        # Save current file position and data
        current_records = []

        # Read all existing records
        self._file.seek(0)
        for i in range(self._size):
            offset = i * RECORD_SIZE
            self._file.seek(offset)
            data = self._file.read(RECORD_SIZE)
            if len(cast(bytes, data)) == RECORD_SIZE:
                record = struct.unpack(STRUCT_FORMAT, cast(Buffer, data))
                current_records.append(OHLCV(*record, extra_fields={}))

        # Create temp file for rebuilding
        temp_fd, temp_path = tempfile.mkstemp(suffix='.ohlcv.tmp', dir=os.path.dirname(self.path))
        try:
            # Close temp file descriptor as we'll open it differently
            os.close(temp_fd)

            # Create new writer with temp file
            with OHLCVWriter(temp_path) as temp_writer:
                # Write all records with correct interval
                # The writer will now properly handle gaps
                for record in current_records:
                    temp_writer.write(record)

            # Close current file
            self._file.close()

            # Replace original with rebuilt file
            shutil.move(temp_path, self.path)

            # Reopen the file
            self._file = open(self.path, 'rb+')
            self._size = os.path.getsize(self.path) // RECORD_SIZE

            # Reset interval to the correct one
            self._interval = new_interval

            # Position at end for appending
            self._file.seek(0, os.SEEK_END)
            self._current_pos = self._size

            # Update last timestamp
            if self._size > 0:
                self._file.seek((self._size - 1) * RECORD_SIZE)
                data: Buffer = self._file.read(4)
                self._last_timestamp = struct.unpack('I', data)[0]
                self._file.seek(0, os.SEEK_END)

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise IOError(f"Failed to rebuild file with correct interval: {e}")

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
        # Parse timezone
        timezone = None
        if tz:
            if tz.startswith(('+', '-')):
                # Handle UTC offset fmt (e.g. +0100, -0500)
                sign = 1 if tz.startswith('+') else -1
                hours = int(tz[1:3])
                minutes = int(tz[3:]) if len(tz) > 3 else 0
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
                    ts_str = row[timestamp_idx]

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

    def load_from_txt(self, path: str | Path,
                      timestamp_format: str | None = None,
                      timestamp_column: str | None = None,
                      date_column: str | None = None,
                      time_column: str | None = None,
                      tz: str | None = None) -> None:
        """
        Load OHLCV data from TXT file using only builtin modules.

        :param path: Path to TXT file
        :param timestamp_format: Optional datetime fmt for parsing
        :param timestamp_column: Column name for timestamp (default tries: timestamp, time, date)
        :param date_column: When timestamp is split into date+time columns, date column name
        :param time_column: When timestamp is split into date+time columns, time column name
        :param tz: Timezone name (e.g. 'UTC', 'Europe/London', '+0100') for timestamp conversion
        """
        # Parse timezone
        timezone = None
        if tz:
            if tz.startswith(('+', '-')):
                # Handle UTC offset fmt (e.g. +0100, -0500)
                sign = 1 if tz.startswith('+') else -1
                hours = int(tz[1:3])
                minutes = int(tz[3:]) if len(tz) > 3 else 0
                timezone = dt_timezone(sign * timedelta(hours=hours, minutes=minutes))
            else:
                # Handle named timezone (e.g. UTC, Europe/London)
                try:
                    timezone = ZoneInfo(tz)
                except Exception as e:
                    raise ValueError(f"Invalid timezone {tz}: {e}")

        # Auto-detect delimiter
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("File is empty or first line is blank")

            # Check for common delimiters in order of preference
            delimiters = ['\t', ';', '|']
            delimiter_counts = {}

            for delim in delimiters:
                count = first_line.count(delim)
                if count > 0:
                    delimiter_counts[delim] = count

            if not delimiter_counts:
                raise ValueError("No supported delimiter found (tab, semicolon, or pipe)")

            # Use delimiter with highest count
            delimiter = max(delimiter_counts, key=lambda x: delimiter_counts[x])

        # Read TXT file with manual parsing for better control
        with open(path, 'r') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("File is empty")

        # Parse header line
        header_line = lines[0].strip()
        if not header_line:
            raise ValueError("Header row is empty")

        headers = self._parse_txt_line(header_line, delimiter)
        headers = [h.lower().strip() for h in headers]  # Case insensitive

        if not headers:
            raise ValueError("No headers found")

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
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            row = self._parse_txt_line(line, delimiter)

            if len(row) != len(headers):
                raise ValueError(f"Row has {len(row)} columns, expected {len(headers)}")

            # Strip whitespace from all fields
            row = [field.strip() for field in row]

            # Handle timestamp
            try:
                if date_idx is not None and time_idx is not None:
                    # Combine date and time
                    ts_str = f"{row[date_idx]} {row[time_idx]}"
                else:
                    ts_str = str(row[timestamp_idx]) if timestamp_idx is not None and timestamp_idx < len(row) else ""

                # Convert timestamp
                if ts_str.isdigit():
                    timestamp = int(ts_str)
                else:
                    dt = None
                    if timestamp_format:
                        dt = datetime.strptime(ts_str, timestamp_format)
                    else:
                        # Try common formats
                        for fmt in [
                            '%Y-%m-%d %H:%M:%S%z',  # 2025-01-08 19:00:00+0000
                            '%Y-%m-%d %H:%M:%S%Z',  # 2025-01-08 19:00:00UTC
                            '%Y-%m-%dT%H:%M:%S%z',  # 2025-01-08T19:00:00+0000
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

                        if dt is None:
                            raise ValueError(f"Could not parse timestamp: {ts_str}")

                    # Set timezone if specified and convert to timestamp
                    if timezone and dt is not None:
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

    @staticmethod
    def _parse_txt_line(line: str, delimiter: str) -> list[str]:
        """
        Parse a single TXT line with proper handling of quoted fields and escape characters.
        
        :param line: Line to parse
        :param delimiter: Delimiter character
        :return: List of parsed fields
        :raises ValueError: If line format is invalid
        """
        if not line:
            return []

        fields = []
        current_field = ""
        in_quotes = False
        quote_char = None
        i = 0

        while i < len(line):
            char = line[i]

            # Handle escape characters
            if char == '\\' and i + 1 < len(line):
                next_char = line[i + 1]
                if next_char in ['"', "'", '\\', 'n', 't', 'r']:
                    if next_char == 'n':
                        current_field += '\n'
                    elif next_char == 't':
                        current_field += '\t'
                    elif next_char == 'r':
                        current_field += '\r'
                    else:
                        current_field += next_char
                    i += 2
                    continue
                else:
                    current_field += char
                    i += 1
                    continue

            # Handle quotes
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                i += 1
                continue
            elif char == quote_char and in_quotes:
                # Check for escaped quote (double quote)
                if i + 1 < len(line) and line[i + 1] == quote_char:
                    current_field += char
                    i += 2
                    continue
                else:
                    in_quotes = False
                    quote_char = None
                    i += 1
                    continue

            # Handle delimiter
            if char == delimiter and not in_quotes:
                fields.append(current_field)
                current_field = ""
                i += 1
                continue

            # Regular character
            current_field += char
            i += 1

        # Add the last field
        fields.append(current_field)

        # Validate that quotes are properly closed
        if in_quotes:
            raise ValueError(f"Unclosed quote in line: {line[:50]}...")

        return fields

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
        # Parse timezone
        timezone = None
        if tz:
            if tz.startswith(('+', '-')):
                # Handle UTC offset format
                sign = 1 if tz.startswith('+') else -1
                hours = int(tz[1:3])
                minutes = int(tz[3:]) if len(tz) > 3 else 0
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
                self._start_timestamp = struct.unpack('I', cast(Buffer, self._mmap[0:4]))[0]
                second_timestamp = struct.unpack('I', cast(Buffer, self._mmap[RECORD_SIZE:RECORD_SIZE + 4]))[0]
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
                # Skip gaps (volume == -1)
                if candle.volume == -1:
                    continue
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
        data = []
        for candle in self:
            # Skip gaps (volume == -1)
            if candle.volume == -1:
                continue
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
