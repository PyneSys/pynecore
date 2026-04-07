"""
Bar Magnifier — groups lower-timeframe OHLCV candles into chart-timeframe windows.

Used by ScriptRunner when use_bar_magnifier=true: the script sees aggregated chart-TF
bars, while the broker emulator processes orders against each sub-bar for accurate fills.
"""
from dataclasses import dataclass
from datetime import timezone as dt_timezone
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

from .aggregator import _merge_candles
from .resampler import Resampler
from ..types.ohlcv import OHLCV

__all__ = ['BarMagnifier', 'MagnifiedWindow']


@dataclass(slots=True)
class MagnifiedWindow:
    """A single chart-timeframe bar with its constituent sub-bars."""
    sub_bars: list[OHLCV]
    aggregated: OHLCV
    is_last_window: bool


class BarMagnifier:
    """
    Groups sub-timeframe OHLCV candles into chart-timeframe windows.

    Uses Resampler for bar boundary alignment (same logic as aggregator.py).
    Yields MagnifiedWindow objects with peek-ahead for last-window detection.
    """

    def __init__(
            self,
            ohlcv_iter: Iterable[OHLCV],
            chart_tf: str,
            tz: ZoneInfo | dt_timezone | None = None,
    ):
        """
        :param ohlcv_iter: Iterator of sub-timeframe OHLCV candles
        :param chart_tf: Chart timeframe string (e.g., '60', '1D')
        :param tz: Timezone for day/week/month boundary alignment
        """
        self._ohlcv_iter = ohlcv_iter
        self._resampler = Resampler.get_resampler(chart_tf)
        self._tz = tz

    def _get_bar_time(self, candle: OHLCV) -> int:
        """Get the chart-bar opening timestamp (seconds) for a sub-bar candle."""
        bar_time_ms = self._resampler.get_bar_time(candle.timestamp * 1000, tz=self._tz)
        return bar_time_ms // 1000

    def __iter__(self) -> Iterator[MagnifiedWindow]:
        window: list[OHLCV] = []
        current_bar_time: int | None = None
        next_window: MagnifiedWindow | None = None

        for candle in self._ohlcv_iter:
            bar_time = self._get_bar_time(candle)

            if current_bar_time is not None and bar_time != current_bar_time:
                # New bar boundary — flush current window
                new_window = MagnifiedWindow(
                    sub_bars=window,
                    aggregated=_merge_candles(window, current_bar_time),
                    is_last_window=False,
                )

                # Peek-ahead: yield the previous window (now we know it's not the last)
                if next_window is not None:
                    yield next_window
                next_window = new_window
                window = []

            current_bar_time = bar_time
            window.append(candle)

        # Flush last window
        if window and current_bar_time is not None:
            last_window = MagnifiedWindow(
                sub_bars=window,
                aggregated=_merge_candles(window, current_bar_time),
                is_last_window=True,
            )

            if next_window is not None:
                yield next_window
            yield last_window
        elif next_window is not None:
            # Edge case: no trailing candles, previous window is the last
            next_window.is_last_window = True
            yield next_window
