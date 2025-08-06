"""
OHLCV data aggregation functionality for converting between timeframes.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from pynecore.types.ohlcv import OHLCV
from pynecore.lib.timeframe import in_seconds
from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter


@dataclass
class AggregationResult:
    """
    Result of aggregation operation.
    :param source_path: Path to source OHLCV file
    :param target_path: Path to target OHLCV file
    :param candles_processed: Number of source candles processed
    :param candles_aggregated: Number of target candles created
    :param start_timestamp: First timestamp in source data
    :param end_timestamp: Last timestamp in source data
    :param duration_seconds: Time taken for aggregation
    """
    source_path: Path
    target_path: Path
    candles_processed: int
    candles_aggregated: int
    start_timestamp: int
    end_timestamp: int
    duration_seconds: float


class TimeframeAggregator:
    """
    Aggregates OHLCV data from lower to higher timeframes.
    """

    def __init__(self, source_timeframe: str, target_timeframe: str):
        """
        Initialize aggregator with source and target timeframes.
        :param source_timeframe: Source timeframe string (e.g., '5', '1H')
        :param target_timeframe: Target timeframe string (e.g., '1H', '1D')
        """
        self.source_timeframe = source_timeframe
        self.target_timeframe = target_timeframe
        self.source_seconds = in_seconds(source_timeframe)
        self.target_seconds = in_seconds(target_timeframe)
        self.window_size = self.target_seconds // self.source_seconds

    def validate_timeframes(self) -> None:
        """
        Validate timeframe compatibility.
        :raises ValueError: If timeframes are incompatible
        """
        if self.target_seconds <= self.source_seconds:
            raise ValueError(
                f"Target timeframe ({self.target_timeframe}) must be larger than "
                f"source timeframe ({self.source_timeframe})"
            )

        if self.target_seconds % self.source_seconds != 0:
            raise ValueError(
                f"Target timeframe ({self.target_timeframe}) must be evenly divisible by "
                f"source timeframe ({self.source_timeframe})"
            )

    @staticmethod
    def aggregate_candles(candles: list[OHLCV]) -> OHLCV:
        """
        Aggregate a list of candles into a single candle.
        :param candles: List of OHLCV candles to aggregate
        :return: Aggregated OHLCV candle
        :raises ValueError: If candles list is empty
        """
        if not candles:
            raise ValueError("Cannot aggregate empty candle list")

        # OHLCV aggregation rules:
        # - Open: First candle's open
        # - High: Maximum of all highs
        # - Low: Minimum of all lows
        # - Close: Last candle's close
        # - Volume: Sum of all volumes
        # - Timestamp: First candle's timestamp (aligned to target timeframe)

        first_candle = candles[0]
        last_candle = candles[-1]

        return OHLCV(
            timestamp=first_candle.timestamp,
            open=first_candle.open,
            high=max(c.high for c in candles),
            low=min(c.low for c in candles),
            close=last_candle.close,
            volume=sum(c.volume for c in candles),
            extra_fields={}
        )

    def aggregate_file(
            self,
            source_path: Path,
            target_path: Path,
            truncate: bool = False
    ) -> AggregationResult:
        """
        Aggregate entire OHLCV file.
        :param source_path: Path to source .ohlcv file
        :param target_path: Path to target .ohlcv file
        :param truncate: Whether to truncate existing target file
        :return: Aggregation result
        :raises ValueError: If timeframes are incompatible
        :raises FileNotFoundError: If source file doesn't exist
        """
        self.validate_timeframes()

        start_time = time.time()
        candles_processed = 0
        candles_aggregated = 0

        with OHLCVReader(source_path) as reader:
            with OHLCVWriter(target_path) as writer:
                if truncate:
                    writer.truncate()

                # Determine starting position
                start_position = 0
                if not truncate and writer.end_timestamp:
                    # Resume from last aggregated timestamp
                    last_target_time = writer.end_timestamp
                    # Find corresponding position in source data
                    start_position = self._find_resume_position(reader, last_target_time)

                # Process data by aggregating candles into windows
                current_window = []
                window_start_time = None

                for i in range(start_position, reader.size):
                    candle = reader.read(i)
                    candles_processed += 1

                    # Align timestamp to target timeframe
                    aligned_timestamp = self._align_timestamp(candle.timestamp)

                    # Start new window if needed
                    if window_start_time is None:
                        window_start_time = aligned_timestamp

                    # Check if candle belongs to current window
                    if aligned_timestamp == window_start_time:
                        current_window.append(candle)
                    else:
                        # Aggregate current window and start new one
                        if current_window:
                            aggregated = self.aggregate_candles(current_window)
                            writer.write(aggregated)
                            candles_aggregated += 1

                        # Start new window
                        current_window = [candle]
                        window_start_time = aligned_timestamp

                # Process final window
                if current_window:
                    aggregated = self.aggregate_candles(current_window)
                    writer.write(aggregated)
                    candles_aggregated += 1

        return AggregationResult(
            source_path=source_path,
            target_path=target_path,
            candles_processed=candles_processed,
            candles_aggregated=candles_aggregated,
            start_timestamp=reader.start_timestamp if reader.size > 0 else 0,
            end_timestamp=reader.end_timestamp if reader.size > 0 else 0,
            duration_seconds=time.time() - start_time
        )

    def _align_timestamp(self, timestamp: int) -> int:
        """
        Align timestamp to target timeframe boundary.
        :param timestamp: Unix timestamp in seconds
        :return: Aligned timestamp in seconds
        """
        # Align timestamp to target timeframe boundary
        aligned_seconds = (timestamp // self.target_seconds) * self.target_seconds
        return aligned_seconds

    @staticmethod
    def _find_resume_position(reader: OHLCVReader, last_target_time: int) -> int:
        """
        Find position in source data to resume aggregation.
        :param reader: OHLCV reader for source data
        :param last_target_time: Last aggregated timestamp
        :return: Index position to resume from
        """
        # Binary search for efficient position finding
        left, right = 0, reader.size - 1

        while left <= right:
            mid = (left + right) // 2
            candle = reader.read(mid)

            if candle.timestamp <= last_target_time:
                left = mid + 1
            else:
                right = mid - 1

        return left
