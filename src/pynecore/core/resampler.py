from datetime import datetime, timedelta, timezone as dt_timezone
from typing import ClassVar
from functools import lru_cache
from zoneinfo import ZoneInfo

from .datetime import parse_timezone
from ..lib import timeframe as tf_module


class Resampler:
    """
    Resampler class for handling different timeframes and calculating bar times.

    This class provides functionality to resample data to different timeframes
    and calculate the opening time of bars for various timeframe specifications.
    """

    _resamplers: ClassVar[dict[str, 'Resampler']] = {}

    def __init__(self, timeframe: str):
        """
        Initialize resampler for a specific timeframe.

        :param timeframe: Timeframe string (e.g., "1D", "4H", "60", "15")
        """
        self.timeframe = timeframe
        self._validate_timeframe()

    def _validate_timeframe(self) -> None:
        """Validate that the timeframe is supported."""
        try:
            tf_module.in_seconds(self.timeframe)
        except (ValueError, AssertionError) as e:
            raise ValueError(f"Invalid timeframe: {self.timeframe}") from e

    @classmethod
    @lru_cache(maxsize=128)
    def get_resampler(cls, timeframe: str) -> 'Resampler':
        """
        Get a resampler instance for the specified timeframe.

        :param timeframe: Timeframe string
        :return: Resampler instance
        :raises ValueError: If timeframe is invalid
        """
        if timeframe not in cls._resamplers:
            cls._resamplers[timeframe] = cls(timeframe)
        return cls._resamplers[timeframe]

    def get_bar_time(self, current_time_ms: int,
                     tz: ZoneInfo | dt_timezone | None = None) -> int:
        """
        Calculate the bar opening time for the current timeframe.

        For seconds and minutes, timezone is irrelevant (pure epoch arithmetic).
        For daily, weekly, and monthly timeframes, the timezone determines where
        midnight falls — i.e., which calendar day a timestamp belongs to.

        :param current_time_ms: Current time in milliseconds (UNIX timestamp)
        :param tz: Timezone for day/week/month boundary calculation.
                   If None, uses the system's local timezone.
        :return: Bar opening time in milliseconds
        """
        # Convert to seconds for calculations
        current_time_sec = current_time_ms // 1000

        # Get timeframe in seconds
        tf_seconds = tf_module.in_seconds(self.timeframe)

        # Calculate bar opening time based on timeframe type
        modifier, multiplier = tf_module._process_tf(self.timeframe)

        if modifier == 'S':  # Seconds
            bar_start_sec = (current_time_sec // tf_seconds) * tf_seconds

        elif modifier == '':  # Minutes
            # Round down to the nearest timeframe boundary
            bar_start_sec = (current_time_sec // tf_seconds) * tf_seconds

        elif modifier in ('D', 'W', 'M'):
            # Daily/Weekly/Monthly — timezone matters for calendar alignment
            if tz is not None:
                current_dt = datetime.fromtimestamp(current_time_sec, tz=tz)
            else:
                current_dt = datetime.fromtimestamp(current_time_sec)

            if modifier == 'D':  # Daily
                bar_start_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)

                # For multi-day timeframes, align to the start of the period
                if multiplier > 1:
                    epoch = datetime(1970, 1, 1, tzinfo=tz)
                    days_since_epoch = (bar_start_dt - epoch).days
                    aligned_days = (days_since_epoch // multiplier) * multiplier
                    bar_start_dt = epoch + timedelta(days=aligned_days)

            elif modifier == 'W':  # Weekly
                bar_start_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                days_to_monday = bar_start_dt.weekday()  # 0 = Monday
                bar_start_dt -= timedelta(days=days_to_monday)

                # For multi-week timeframes, align to the start of the period
                if multiplier > 1:
                    epoch_monday = datetime(1970, 1, 5, tzinfo=tz)
                    weeks_since_epoch = (bar_start_dt - epoch_monday).days // 7
                    aligned_weeks = (weeks_since_epoch // multiplier) * multiplier
                    bar_start_dt = epoch_monday + timedelta(weeks=aligned_weeks)

            else:  # Monthly
                bar_start_dt = current_dt.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0)

                # For multi-month timeframes, align to the start of the period
                if multiplier > 1:
                    months_since_reference = (
                        (bar_start_dt.year - 1970) * 12 + (bar_start_dt.month - 1))
                    aligned_months = (
                        (months_since_reference // multiplier) * multiplier)
                    target_year = 1970 + aligned_months // 12
                    target_month = (aligned_months % 12) + 1
                    bar_start_dt = bar_start_dt.replace(
                        year=target_year, month=target_month)

            bar_start_sec = int(bar_start_dt.timestamp())

        else:
            raise ValueError(f"Unsupported timeframe modifier: {modifier}")

        # Convert back to milliseconds
        return bar_start_sec * 1000