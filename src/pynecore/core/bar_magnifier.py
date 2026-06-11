"""
Bar Magnifier — groups lower-timeframe OHLCV candles into chart-timeframe windows.

Used by ScriptRunner when use_bar_magnifier=true: the script sees aggregated chart-TF
bars, while the broker emulator processes orders against each sub-bar for accurate fills.

Multi-period chart timeframes (nD/nW/nM, n > 1) live on the year-reset scheduled
grid (see the ``resampler`` module docs); 'observed' symbols (exchange-listed)
count the actual trading days seen in the sub-bar stream, which reproduces
TradingView's holiday-aware grid.
"""
from dataclasses import dataclass
from datetime import timedelta, timezone as dt_timezone
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

from .aggregator import _merge_candles
from .resampler import (
    Resampler, ObservedDayCounter, grid_mode, overnight_opens, trading_day,
    first_monday,
)
from ..lib.timeframe import in_seconds, _process_tf
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

    Uses Resampler for bar boundary alignment (same logic as aggregator.py);
    multi-period (nD/nW/nM) chart timeframes on 'observed' symbols count the
    actual trading days in the stream instead. Yields MagnifiedWindow objects
    with peek-ahead for last-window detection.
    """

    def __init__(
            self,
            ohlcv_iter: Iterable[OHLCV],
            chart_tf: str,
            tz: ZoneInfo | dt_timezone | None = None,
            session_starts: 'list | None' = None,
            opening_hours: 'list | None' = None,
            sym_type: str | None = None,
            source_tf: str | None = None,
    ):
        """
        :param ohlcv_iter: Iterator of sub-timeframe OHLCV candles
        :param chart_tf: Chart timeframe string (e.g., '60', '1D')
        :param tz: Timezone for day/week/month boundary alignment
        :param session_starts: Per-trading-day primary opens for intraday session
            anchoring and multi-period grids. ``None`` keeps the pure clock-floor
            (see :meth:`Resampler.get_bar_time`).
        :param opening_hours: ``SymInfo.opening_hours`` — trading-day roll source
            for multi-period (nD/nW/nM) grids.
        :param sym_type: ``SymInfo.type`` for :func:`grid_mode` classification of
            multi-period grids.
        :param source_tf: Sub-bar timeframe string. On multi-period chart
            timeframes an intraday sub-bar belongs to the trading day its *last*
            instant falls into — the bar containing a session open starts the
            new day even when its timestamp precedes the open.
        """
        self._ohlcv_iter = ohlcv_iter
        self._resampler = Resampler.get_resampler(chart_tf)
        self._tz = tz
        self._session_starts = session_starts
        self._opening_hours = opening_hours

        # noinspection PyProtectedMember
        modifier, multiplier = _process_tf(chart_tf)
        self._modifier = modifier
        self._multiplier = multiplier
        multi = modifier in ('D', 'W', 'M') and multiplier > 1
        self._mode = grid_mode(sym_type, opening_hours) if multi else None

        self._src_off = 0
        if multi and source_tf:
            # noinspection PyProtectedMember
            src_mod, _ = _process_tf(source_tf)
            if src_mod in ('', 'S'):
                self._src_off = in_seconds(source_tf) - 1

        if multi and self._mode == 'observed':
            self._overnight = overnight_opens(opening_hours, session_starts)
            self._counter: ObservedDayCounter | None = ObservedDayCounter()
        else:
            self._overnight = {}
            self._counter = None

    def _key_and_stamp(self, candle: OHLCV) -> tuple[object, int]:
        """
        Grouping key and window-opening timestamp (seconds) for a sub-bar.

        For 'observed' multi-period grids the key counts the actual trading
        days and the window is stamped by its first sub-bar; everything else
        uses the scheduled-grid bar time as both.
        """
        if self._counter is not None:
            td = trading_day(candle.timestamp + self._src_off, self._tz, self._overnight)
            if self._modifier == 'D':
                key = (td.year, self._counter.ordinal(td) // self._multiplier)
            elif self._modifier == 'W':
                monday = td - timedelta(days=td.weekday())
                weeks = (monday - first_monday(monday.year)).days // 7
                key = (monday.year, weeks // self._multiplier)
            else:  # 'M'
                key = (td.year, (td.month - 1) // self._multiplier)
            return key, candle.timestamp

        bar_time_ms = self._resampler.get_bar_time(
            (candle.timestamp + self._src_off) * 1000, tz=self._tz,
            session_starts=self._session_starts,
            opening_hours=self._opening_hours, mode=self._mode)
        bar_time = bar_time_ms // 1000
        return bar_time, bar_time

    def __iter__(self) -> Iterator[MagnifiedWindow]:
        window: list[OHLCV] = []
        current_key: object | None = None
        window_stamp: int | None = None
        next_window: MagnifiedWindow | None = None

        for candle in self._ohlcv_iter:
            key, stamp = self._key_and_stamp(candle)

            if current_key is not None and key != current_key:
                # New bar boundary — flush current window
                new_window = MagnifiedWindow(
                    sub_bars=window,
                    aggregated=_merge_candles(window, window_stamp),
                    is_last_window=False,
                )

                # Peek-ahead: yield the previous window (now we know it's not the last)
                if next_window is not None:
                    yield next_window
                next_window = new_window
                window = []
                window_stamp = None

            current_key = key
            if window_stamp is None:
                window_stamp = stamp
            window.append(candle)

        # Flush last window
        if window and window_stamp is not None:
            last_window = MagnifiedWindow(
                sub_bars=window,
                aggregated=_merge_candles(window, window_stamp),
                is_last_window=True,
            )

            if next_window is not None:
                yield next_window
            yield last_window
        elif next_window is not None:
            # Edge case: no trailing candles, previous window is the last
            next_window.is_last_window = True
            yield next_window
