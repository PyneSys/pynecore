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

A period closes following TradingView's ``lookahead_off`` merge rule: the
confirmed chart bar whose close instant reaches the HTF period end completes
the bar, so the closed HTF bar is delivered together with the period's LAST
chart bar — not with the next period's first one. When no such completing bar
arrives (data gap, or the chart bar span is unknown — D/W/M charts), the close
falls back to the first chart bar that crosses into a new HTF period.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..lib import timeframe as tf_module
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

    ``exact`` is ``False`` when the folded chart bars do not cover the period
    from its start to its end without a hole: the first one opens after the
    period start (the chart starts inside the period, has a gap there, or its
    previous bar ran into this period), a chart bar inside the period is
    missing, the period was left without its completing bar, or a chart bar runs
    past the period end. On a D/W/M chart (no chart bar span) the coverage cannot
    be proven, so every closed fold is inexact there. Such a fold is not the HTF
    bar, so a closed bar marked inexact has to come from the context's own feed.
    """
    period_start: int  # HTF period open time in ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    last_chart_time_ms: int = 0
    last_chart_volume: float = 0.0
    exact: bool = True

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

    __slots__ = ('_timeframe', '_tz', '_resampler', '_state', '_session_starts',
                 '_chart_span_ms', '_opening_hours', '_grid_mode', '_period_ms',
                 'also_closed')

    def __init__(self, timeframe: str, tz: 'ZoneInfo',
                 session_starts: 'list | None' = None,
                 chart_span_ms: int = 0,
                 opening_hours: 'list | None' = None,
                 grid_mode: str | None = None):
        self._timeframe = timeframe
        self._tz = tz
        self._resampler = Resampler.get_resampler(timeframe)
        self._state: DevelopingBar | None = None
        # Intraday session anchoring (TradingView aligns HTF bars to the session
        # open). ``None`` keeps the pure clock-floor — see ``Resampler.get_bar_time``.
        self._session_starts = session_starts
        # Trading-day roll and scheduled-day calendar of the multi-period
        # (nD/nW/nM) grid. ``None`` lets the resampler infer both.
        self._opening_hours = opening_hours
        self._grid_mode = grid_mode
        # Chart bar span in ms for the close-instant period completion check.
        # 0 (D/W/M charts: no fixed arithmetic span) disables it — the period
        # then closes on the first chart bar of the next period instead.
        self._chart_span_ms = chart_span_ms
        # noinspection PyProtectedMember
        self._period_ms = tf_module._in_seconds(timeframe) * 1000
        #: With ``double_close``: the period this chart bar completed in the same
        #: update that delivered the previous, gap-left period as ``closed``.
        self.also_closed: DevelopingBar | None = None

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def period_ms(self) -> int:
        """Nominal HTF period length in ms (a session can cut a period short)."""
        return self._period_ms

    @property
    def current(self) -> DevelopingBar | None:
        """Currently developing HTF bar, or None if no chart bar seen yet."""
        return self._state

    def update(
        self,
        chart_time_ms: int,
        chart_open: float, chart_high: float, chart_low: float,
        chart_close: float, chart_volume: float,
        chart_confirmed: bool = True,
        period_complete: bool | None = None,
        double_close: bool = False,
    ) -> tuple[bool, DevelopingBar | None, DevelopingBar | None]:
        """
        Fold one chart bar into the developing HTF bar.

        :param chart_time_ms: Chart bar open timestamp in ms.
        :param chart_open: Chart bar open price.
        :param chart_high: Chart bar high price.
        :param chart_low: Chart bar low price.
        :param chart_close: Chart bar close price.
        :param chart_volume: Chart bar volume (cumulative within the bar for
            live providers; the aggregator deduplicates same-bar updates).
        :param chart_confirmed: Whether the folded chart bar is final
            (``barstate.isconfirmed``). Only a confirmed chart bar may complete
            the period — a developing intra-bar tick must not freeze a
            non-final HTF close.
        :param period_complete: Overrides the built-in close-instant rule with a
            caller-computed decision — ``True`` when this chart bar's own close
            reaches the HTF period's close instant. Used by the script-level
            timeframe path, which resolves both closes from the trading
            schedule (see :mod:`pynecore.core.script_timeframe`).
        :param double_close: When this chart bar both closes the previous,
            gap-left period and completes its own, complete its own period too:
            it is returned as ``also_closed`` and no developing bar is left.
            Without it the own period closes one chart bar late, via the
            fallback of the next update.
        :return: ``(is_new_period, developing_or_none, closed_or_none)``

            - ``is_new_period`` — True iff the chart bar opens a fresh HTF period.
            - ``developing_or_none`` — the accumulated HTF bar including this
              chart bar, or None when this chart bar just completed the period
              (no fresh developing bar exists until the next chart bar).
            - ``closed_or_none`` — the completed period's final accumulated bar:
              delivered with the confirmed chart bar whose close instant reaches
              the period end (TV ``lookahead_off`` merge rule), or — when no
              such bar arrived — with the first chart bar of the next period.
        """
        period_start = self._resampler.get_bar_time(
            chart_time_ms, self._tz, self._session_starts,
            self._opening_hours, self._grid_mode)

        closed: DevelopingBar | None = None
        self.also_closed = None
        state = self._state
        if state is None or period_start != state.period_start:
            is_new = True
            if state is not None:
                # The period was left without its completing chart bar (data
                # gap) — deliver its accumulated final form one bar late. With
                # a known chart bar span that bar is missing from the fold.
                # Without one (D/W/M charts) this is how every period closes,
                # and the fold's coverage cannot be proven: a weekly bar opening
                # on September 29 carries October's first days into September,
                # and a missing daily bar leaves no trace. Inexact either way.
                closed = state
                closed.exact = False
            state = DevelopingBar(
                period_start=period_start,
                open=chart_open, high=chart_high, low=chart_low,
                close=chart_close, volume=chart_volume,
                last_chart_time_ms=chart_time_ms,
                last_chart_volume=chart_volume,
                # The fold misses the period's head unless this bar opens it.
                exact=chart_time_ms == period_start,
            )
            self._state = state
        else:
            is_new = False
            if (self._chart_span_ms and chart_time_ms != state.last_chart_time_ms
                    and chart_time_ms != state.last_chart_time_ms + self._chart_span_ms):
                # A chart bar inside the period is missing from the fold.
                state.exact = False
            state.update(chart_time_ms, chart_high, chart_low,
                         chart_close, chart_volume)

        # TV ``lookahead_off`` merge rule: the confirmed chart bar whose close
        # instant reaches the period end completes the HTF bar — deliver it
        # now, one chart bar earlier than the next-period fallback above.
        # Skipped when a gap-closed bar is already pending, unless the caller
        # takes both (``double_close``); otherwise the fresh period then closes
        # via the fallback. Also skipped when the chart bar span is unknown
        # (``chart_span_ms == 0``).
        if period_complete is None:
            period_complete = bool(self._chart_span_ms) and self._resampler.get_bar_time(
                chart_time_ms + self._chart_span_ms, self._tz,
                self._session_starts, self._opening_hours,
                self._grid_mode) != period_start
            # A bar that completes the period may also run PAST its end: its last
            # millisecond lies in a later period (a 3-minute bar opening at 02:03
            # under a 5-minute period ends at 02:06). Only a bar that reaches the
            # period end can, so the extra lookup is paid on those alone.
            if period_complete and self._resampler.get_bar_time(
                    chart_time_ms + self._chart_span_ms - 1, self._tz,
                    self._session_starts, self._opening_hours,
                    self._grid_mode) != period_start:
                state.exact = False
        if chart_confirmed and period_complete:
            if closed is None:
                self._state = None
                return is_new, None, state
            if double_close:
                self._state = None
                self.also_closed = state
                return is_new, None, closed

        return is_new, state, closed

    def reset(self) -> None:
        """Discard any in-flight developing bar (used on context teardown)."""
        self._state = None
        self.also_closed = None
