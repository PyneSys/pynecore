"""
OHLCV timeframe aggregation — converts lower timeframe data to higher timeframes.

Uses Resampler.get_bar_time() for correct bar boundary alignment across all
timeframe types including weekly and monthly.
"""
from datetime import timezone as dt_timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .ohlcv_file import OHLCVReader, OHLCVWriter
from .resampler import Resampler
from ..lib.timeframe import in_seconds
from ..types.ohlcv import OHLCV


def validate_aggregation(source_tf: str, target_tf: str) -> None:
    """
    Validate that aggregation from source to target timeframe is possible.

    :param source_tf: Source timeframe string (e.g., '5', '1D')
    :param target_tf: Target timeframe string (e.g., '60', '1W')
    :raises ValueError: If timeframes are incompatible
    """
    source_sec = in_seconds(source_tf)
    target_sec = in_seconds(target_tf)

    if target_sec <= source_sec:
        raise ValueError(
            f"Target timeframe ({target_tf}) must be larger than "
            f"source timeframe ({source_tf})"
        )

    if target_sec % source_sec != 0:
        raise ValueError(
            f"Target timeframe ({target_tf}) must be evenly divisible by "
            f"source timeframe ({source_tf})"
        )


def _merge_candles(candles: list[OHLCV], bar_time: int) -> OHLCV:
    """
    Merge a window of candles into a single aggregated candle.

    :param candles: Non-empty list of OHLCV candles belonging to the same bar
    :param bar_time: Aligned bar opening timestamp in seconds
    :return: Aggregated OHLCV candle
    """
    return OHLCV(
        timestamp=bar_time,
        open=candles[0].open,
        high=max(c.high for c in candles),
        low=min(c.low for c in candles),
        close=candles[-1].close,
        volume=sum(c.volume for c in candles),
    )


def aggregate_ohlcv(
        source_path: Path,
        target_path: Path,
        target_tf: str,
        tz: ZoneInfo | dt_timezone | None = None,
) -> tuple[int, int]:
    """
    Aggregate OHLCV data from a lower timeframe file to a higher timeframe file.

    :param source_path: Path to source .ohlcv file
    :param target_path: Path to target .ohlcv file (will be overwritten)
    :param target_tf: Target timeframe string (e.g., '60', '1W')
    :param tz: Timezone for day/week/month boundary alignment.
               Should match the data's timezone (from TOML metadata).
    :return: Tuple of (source_candles_read, target_candles_written)
    """
    resampler = Resampler.get_resampler(target_tf)

    source_count = 0
    target_count = 0

    with OHLCVReader(source_path) as reader:
        with OHLCVWriter(target_path, truncate=True) as writer:
            window: list[OHLCV] = []
            current_bar_time: int | None = None

            start_ts = reader.start_timestamp
            if start_ts is None:
                return 0, 0

            for candle in reader.read_from(start_ts):
                source_count += 1

                # Resampler works in ms, OHLCV timestamps are in seconds
                bar_time_ms = resampler.get_bar_time(candle.timestamp * 1000, tz=tz)
                bar_time = bar_time_ms // 1000

                if current_bar_time is not None and bar_time != current_bar_time:
                    # New bar boundary — flush the window
                    writer.write(_merge_candles(window, current_bar_time))
                    target_count += 1
                    window = []

                current_bar_time = bar_time
                window.append(candle)

            # Flush last window
            if window and current_bar_time is not None:
                writer.write(_merge_candles(window, current_bar_time))
                target_count += 1

    return source_count, target_count
