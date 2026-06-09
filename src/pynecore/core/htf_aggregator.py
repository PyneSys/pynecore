"""
Higher-timeframe bar aggregator for live ``request.security()`` developing bars.

The chart process aggregates its own (lower-TF) OHLCV stream into a developing
HTF bar for each ``lookahead_on`` security context. The aggregated developing
bar is then transported to the security child via the SyncBlock so the child
can run its ``main()`` function under ``barstate.isconfirmed=False`` using the
chart-derived OHLCV — mirroring the TradingView live ``lookahead_on`` semantics.

Aggregation rules per HTF period (chart bars whose ``get_bar_time(chart_ts)``
maps to the same HTF open):

    open    = first chart bar's open
    high    = max chart high seen so far
    low     = min chart low seen so far
    close   = latest chart close
    volume  = sum of chart volumes (one contribution per chart bar)

Live providers (e.g. CCXT ``watch_ohlcv``) emit repeated intra-bar updates
carrying the *cumulative* candle volume. ``update()`` is therefore keyed on
``chart_time_ms``: when the same chart bar fires again, the prior
contribution of that bar is subtracted from the HTF developing volume and
replaced with the latest cumulative reading — preventing double counting.

When a chart bar crosses into a new HTF period, the previously-accumulated bar
is the *closed* HTF bar for the period just ended, and a fresh developing bar
starts from the new chart bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .resampler import Resampler

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo


@dataclass
class DevelopingBar:
    """In-progress aggregated OHLCV for one HTF period.

    ``last_chart_time_ms`` / ``last_chart_volume`` track the most recently
    folded chart bar so repeated intra-bar updates from a live provider
    (which carry the running cumulative candle volume) can replace their
    own prior contribution instead of stacking it.
    """
    period_start: int  # HTF period open time in ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    last_chart_time_ms: int = 0
    last_chart_volume: float = 0.0

    def update(self, chart_time_ms: int, chart_high: float, chart_low: float,
               chart_close: float, chart_volume: float) -> None:
        if chart_high > self.high:
            self.high = chart_high
        if chart_low < self.low:
            self.low = chart_low
        self.close = chart_close
        if chart_time_ms == self.last_chart_time_ms:
            # Same chart bar re-firing (intra-bar tick): live providers send
            # the running cumulative candle volume, so the previous reading
            # must be subtracted before adding the new one.
            self.volume += chart_volume - self.last_chart_volume
        else:
            self.volume += chart_volume
            self.last_chart_time_ms = chart_time_ms
        self.last_chart_volume = chart_volume


class HTFAggregator:
    """Per-sec_id aggregator of chart OHLCV into a developing HTF bar."""

    __slots__ = ('_timeframe', '_tz', '_resampler', '_state', '_session_starts')

    def __init__(self, timeframe: str, tz: 'ZoneInfo',
                 session_starts: 'list | None' = None):
        self._timeframe = timeframe
        self._tz = tz
        self._resampler = Resampler.get_resampler(timeframe)
        self._state: DevelopingBar | None = None
        # Intraday session anchoring (TradingView aligns HTF bars to the session
        # open). ``None`` keeps the pure clock-floor — see ``Resampler.get_bar_time``.
        self._session_starts = session_starts

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def current(self) -> DevelopingBar | None:
        """Currently developing HTF bar, or None if no chart bar seen yet."""
        return self._state

    def update(
        self,
        chart_time_ms: int,
        chart_open: float, chart_high: float, chart_low: float,
        chart_close: float, chart_volume: float,
    ) -> tuple[bool, DevelopingBar, DevelopingBar | None]:
        """
        Fold one chart bar into the developing HTF bar.

        :param chart_time_ms: Chart bar open timestamp in ms.
        :param chart_open: Chart bar open price.
        :param chart_high: Chart bar high price.
        :param chart_low: Chart bar low price.
        :param chart_close: Chart bar close price.
        :param chart_volume: Chart bar volume (cumulative within the bar for
            live providers; the aggregator deduplicates same-bar updates).
        :return: ``(is_new_period, developing, closed_or_none)``

            - ``is_new_period`` — True iff the chart bar opens a fresh HTF period.
            - ``developing`` — the (now possibly fresh) accumulated HTF bar
              including this chart bar.
            - ``closed_or_none`` — the previous period's final accumulated bar
              when ``is_new_period`` is True, else None.
        """
        period_start = self._resampler.get_bar_time(
            chart_time_ms, self._tz, self._session_starts)

        if self._state is None:
            fresh = DevelopingBar(
                period_start=period_start,
                open=chart_open, high=chart_high, low=chart_low,
                close=chart_close, volume=chart_volume,
                last_chart_time_ms=chart_time_ms,
                last_chart_volume=chart_volume,
            )
            self._state = fresh
            return True, fresh, None

        if period_start != self._state.period_start:
            closed = self._state
            fresh = DevelopingBar(
                period_start=period_start,
                open=chart_open, high=chart_high, low=chart_low,
                close=chart_close, volume=chart_volume,
                last_chart_time_ms=chart_time_ms,
                last_chart_volume=chart_volume,
            )
            self._state = fresh
            return True, fresh, closed

        self._state.update(chart_time_ms, chart_high, chart_low,
                           chart_close, chart_volume)
        return False, self._state, None

    def reset(self) -> None:
        """Discard any in-flight developing bar (used on context teardown)."""
        self._state = None
