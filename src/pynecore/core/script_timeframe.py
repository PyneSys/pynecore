"""
Script-level higher-timeframe execution (``indicator(..., timeframe=...)``).

A script whose declaration carries a ``timeframe`` argument does not run on the
chart bars: it is evaluated in the context of that timeframe and the result is
mapped back onto the chart. This module owns that mapping for the backtest
runner.

``main()`` is evaluated once per COMPLETED higher-timeframe (HTF) bar, built
from the chart feed by :class:`~pynecore.core.htf_aggregator.HTFAggregator`
(open = first chart open, high/low = running extremes, close = last chart close,
volume = sum, time = the HTF period's own open time). The execution lands on the
chart bar whose own close instant reaches the HTF period's close instant -- the
same merge rule ``request.security(..., lookahead_off)`` uses, so an HTF bar's
value never reaches a chart bar that precedes its close. A developing
(incomplete) HTF period produces nothing.

``timeframe_gaps=True`` (Pine's default) plots ``na`` on every chart bar inside
an open HTF period; ``timeframe_gaps=False`` repeats the last confirmed HTF
value on those bars. Series state -- ``bar_index``, history (``[1]``), ``ta.*``
machines, ``var`` variables, strategy position -- advances once per HTF bar,
because the body runs once per HTF bar. ``bar_index`` counts HTF bars from 0,
``last_bar_index`` is the index of the last HTF bar that completes inside the
chart feed and ``last_bar_time`` its open time; both are resolved up front by
:meth:`ScriptTimeframe.prepare`.

The chart timeframe stays the data grid: ``syminfo.period`` keeps reporting it,
while every ``timeframe.*`` builtin reports the script timeframe (see
``lib/timeframe.py::_current_period``), which is what makes an inner
``request.security(syminfo.tickerid, timeframe.period, x)`` a same-context no-op.

Both close instants come from the trading schedule through
:func:`~pynecore.core.security.actual_bar_close` /
:func:`~pynecore.core.security.dwm_session_close`, never from a nominal period
length: a session-bounded daily bar closes when its last session ends (not at
midnight) and a D/W/M chart bar closes at its own civil period end, so neither a
session-bounded symbol nor a D/W/M chart drops its final complete HTF bar.

A chart feed whose bars never reach an HTF period close (a single bar, or a feed
shorter than one HTF period) completes no HTF bar: the run then executes nothing
and writes no output row. That is the correct mapping, not a stall. Only a feed
whose END IS UNKNOWN (a live stream) closes its open period on the last bar, so
the run still has a last bar.

The chart grid must NEST inside the script grid: every chart bar has to fall
entirely inside one HTF period, because an indivisible chart candle cannot be
split between two of them. A 45-minute chart under a 60-minute script, or a
weekly chart under a monthly one, is refused instead of silently folding a
straddling candle into the period its open falls in.
"""
from typing import TYPE_CHECKING, Iterable

from ..types.ohlcv import OHLCV
from ..types.na import na_float
from ..lib import timeframe as tf_module
from .htf_aggregator import HTFAggregator
from .resampler import Resampler, grid_mode
from .security import BarCalendar, actual_bar_close, resolve_session_anchor

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo
    from .syminfo import SymInfo

__all__ = ['ScriptTimeframe']

#: Upper bound on the grid walk of :meth:`ScriptTimeframe.prepare`. One period is
#: counted per step, so this only trips on a feed spanning millions of HTF bars.
_MAX_PERIODS = 10_000_000

#: Seconds in a day -- the nesting divisor of an intraday chart under a D/W/M script.
_DAY_SEC = 86_400

#: Chart -> script D/W/M modifier pairs whose grids nest. A weekly chart does not
#: nest inside a monthly script: a week can straddle two months.
_NESTING_DWM_PAIRS = frozenset({('D', 'D'), ('D', 'W'), ('D', 'M'),
                                ('W', 'W'), ('M', 'M')})


# noinspection PyProtectedMember
class ScriptTimeframe:
    """
    Chart bars -> higher-timeframe bars for a script-level ``timeframe=``.

    One instance per run. :meth:`feed` is called with every chart bar in order
    and answers with the HTF bar to execute on, or ``None`` when the chart bar
    falls inside an open HTF period.
    """

    __slots__ = ('timeframe', 'gaps', 'last_bar_index', 'last_bar_time',
                 '_aggregator', '_resampler', '_tz', '_session_starts',
                 '_chart_timeframe', '_chart_span_ms', '_calendar',
                 '_opening_hours', '_grid_mode', '_feed_end_known',
                 '_last_period', '_closes', '_values', '_na_values', '_is_last')

    def __init__(self, timeframe: str, gaps: bool, chart_timeframe: str,
                 syminfo: 'SymInfo', tz: 'ZoneInfo'):
        """
        :param timeframe: The script's declared timeframe.
        :param gaps: Pine's ``timeframe_gaps`` -- ``True`` leaves the chart bars
            of an open HTF period at ``na``, ``False`` forward-fills them.
        :param chart_timeframe: The chart (data feed) timeframe.
        :param syminfo: Chart symbol info, for session anchoring.
        :param tz: Exchange timezone.
        :raises ValueError: If the script timeframe is finer than the chart's, or
            if the chart grid does not nest inside the script's.
        """
        script_mod, _ = tf_module._process_tf(timeframe)
        chart_mod, chart_mult = tf_module._process_tf(chart_timeframe)
        script_sec = tf_module._in_seconds(timeframe)
        chart_sec = tf_module._in_seconds(chart_timeframe)
        if script_sec < chart_sec:
            # A script timeframe FINER than the chart's is a real feature (a
            # probe on CAPITALCOM:GOLD@60 with ``timeframe='15'`` runs the body
            # on the 15-minute grid -- bar_index steps by 4 per chart bar,
            # timeframe.period answers 900s and the chart bar shows the LAST
            # intrabar execution), but it needs an intrabar feed the runner does
            # not have here. Refused rather than silently run on chart bars.
            raise ValueError(
                f"Script timeframe '{timeframe}' is lower than the chart "
                f"timeframe '{chart_timeframe}'. Running a script on a finer "
                f"grid than the chart feed is not supported; load the data on "
                f"the '{timeframe}' timeframe instead."
            )

        if chart_mod in ('', 'S'):
            nests = (script_sec % chart_sec == 0 if script_mod in ('', 'S')
                     else _DAY_SEC % chart_sec == 0)
        else:
            nests = (chart_mult == 1
                     and (chart_mod, script_mod) in _NESTING_DWM_PAIRS)
        if not nests:
            # A chart candle is indivisible: folding one that straddles two HTF
            # periods into the period of its open silently corrupts both bars
            # (the first gets foreign volume and a close from past its own end,
            # the second loses that stretch entirely).
            raise ValueError(
                f"Chart timeframe '{chart_timeframe}' does not nest inside "
                f"script timeframe '{timeframe}': its bars would straddle the "
                f"script's period boundaries. Load the data on a timeframe the "
                f"script timeframe is a whole multiple of."
            )

        self.timeframe = timeframe
        self.gaps = gaps
        self._chart_timeframe = chart_timeframe
        self._resampler = Resampler.get_resampler(timeframe)
        self._tz = tz
        # Session anchoring (HTF bars align to the session open), decided exactly
        # as a ``request.security()`` HTF context decides it.
        self._session_starts, _, self._opening_hours = resolve_session_anchor(
            syminfo, timeframe, tz)
        # The scheduled-trading-day calendar of the multi-period (nD/nW/nM) grid.
        # Passed explicitly: without it the resampler infers it from the session
        # template, which a 24/7 market that needs no anchoring does not supply,
        # and the period grid would silently fall back to weekdays.
        self._grid_mode = grid_mode(syminfo.type, syminfo.opening_hours)
        self._calendar = BarCalendar(
            tz=tz,
            opening_hours=tuple(syminfo.opening_hours or ()),
            session_starts=tuple(self._session_starts or syminfo.session_starts or ()),
            corrections=getattr(syminfo, 'session_corrections', None) or None,
            grid_mode=self._grid_mode,
        )
        # Nominal chart bar span; 0 for a D/W/M chart, which has none. Only a
        # cheap pre-filter in :meth:`feed` -- the decision is the real close.
        self._chart_span_ms = chart_sec * 1000 if chart_mod in ('', 'S') else 0
        self._aggregator = HTFAggregator(
            timeframe, tz, session_starts=self._session_starts,
            opening_hours=self._opening_hours, grid_mode=self._grid_mode)

        #: Index / open time of the last HTF bar completing inside the feed.
        #: ``None`` until :meth:`prepare` ran (or when the feed end is unknown,
        #: e.g. a live stream) -- callers then fall back to the running index.
        self.last_bar_index: int | None = None
        self.last_bar_time: int | None = None
        self._last_period: int | None = None
        #: Whether :meth:`prepare` was given the feed's end. A known-end feed
        #: never forces an incomplete period to complete; an unknown-end one does.
        self._feed_end_known = False
        self._is_last = False
        #: Memoised HTF period open -> close instant.
        self._closes: dict[int, int] = {}
        # Last executed bar's plot values, and their all-``na`` twin. Both keep
        # the plot column order of the run, which the CSV header is built from.
        self._values: dict[str, float] | None = None
        self._na_values: dict[str, float] | None = None

    # === Grid =============================================================

    def period_start(self, time_ms: int) -> int:
        """
        HTF period a chart timestamp belongs to.

        :param time_ms: Chart bar open time in milliseconds.
        :return: The HTF period's open time in milliseconds.
        """
        return self._resampler.get_bar_time(time_ms, self._tz, self._session_starts,
                                            self._opening_hours, self._grid_mode)

    def _period_close(self, period_start: int) -> int:
        """
        Close instant of the HTF period opening at ``period_start``.

        :param period_start: The HTF period's open time in milliseconds.
        :return: The period's exclusive close instant in milliseconds.
        """
        close = self._closes.get(period_start)
        if close is None:
            close = actual_bar_close(period_start, 0, self._calendar, self.timeframe)
            self._closes[period_start] = close
        return close

    def _chart_close(self, chart_time_ms: int) -> int:
        """
        Close instant of the chart bar opening at ``chart_time_ms``.

        :param chart_time_ms: Chart bar open time in milliseconds.
        :return: The chart bar's exclusive close instant in milliseconds.
        """
        return actual_bar_close(chart_time_ms, 0, self._calendar, self._chart_timeframe)

    def prepare(self, first_time_ms: int, last_time_ms: int | None,
                chart_times: 'Iterable[int] | None' = None) -> None:
        """
        Derive ``last_bar_index`` / ``last_bar_time`` from the chart's span.

        A historical run knows its whole future up front, so both are known
        before the first bar -- but only the CHART's end is known here, and the
        script's last bar is the last HTF period that *completes* inside it.

        :param first_time_ms: First chart bar's open time in milliseconds.
        :param last_time_ms: Last chart bar's open time, or ``None`` when the
            feed end is not known up front (live). Both attributes then stay
            ``None`` and :meth:`feed` falls back to the feed-exhausted rule.
        :param chart_times: Open times of the chart bars that will be fed, when
            the runner can replay them cheaply. Given, the HTF bars are COUNTED
            from the periods actually present; without it the count walks the
            scheduled grid, which over-counts a period the feed has no bar in
            (a market holiday, a data gap).
        :raises ValueError: If the grid walk exceeds :data:`_MAX_PERIODS`.
        """
        if last_time_ms is None:
            return
        self._feed_end_known = True
        first_period = self.period_start(first_time_ms)
        last_period = self.period_start(last_time_ms)
        if self._chart_close(last_time_ms) < self._period_close(last_period):
            # The chart ends mid-period: that HTF bar never completes, so the
            # script's last bar is the one before it.
            last_period = self.period_start(last_period - 1)
        if last_period < first_period:
            return                      # no HTF bar completes inside the feed
        self._last_period = last_period
        self.last_bar_time = last_period

        if chart_times is not None:
            count = -1
            period = -1
            for time_ms in chart_times:
                start = self.period_start(time_ms)
                if start != period:
                    if start > last_period:
                        break
                    period = start
                    count += 1
            self.last_bar_index = max(count, 0)
            if period >= 0:
                # The last period the feed really HAS a bar in -- the scheduled
                # ``last_period`` above may be one the feed skips entirely (a
                # holiday, a data gap), and then no execution ever reaches it,
                # so ``islast`` would never fire.
                self._last_period = period
                self.last_bar_time = period
            return

        # No replayable feed: walk the grid from the first period to the last,
        # counting periods. The probe advances by a fraction of the nominal
        # period length so a civil period shorter than the nominal one
        # (February, a DST week) is never stepped over.
        step = max(1, tf_module._in_seconds(self.timeframe) * 1000 // 8)
        period = first_period
        count = 0
        probe = period
        while period < last_period:
            probe += step
            nxt = self.period_start(probe)
            if nxt != period:
                period = nxt
                count += 1
                if count > _MAX_PERIODS:
                    raise ValueError(
                        f"Script timeframe '{self.timeframe}' spans more than "
                        f"{_MAX_PERIODS} bars over the chart's range."
                    )
        self.last_bar_index = count

    # === Per chart bar ====================================================

    def feed(self, candle: OHLCV, last_chart_bar: bool = False) -> OHLCV | None:
        """
        Fold one chart bar into the developing HTF bar.

        :param candle: The chart bar.
        :param last_chart_bar: Whether this is the final bar of the feed. Only
            used when the feed end was UNKNOWN to :meth:`prepare` (a live
            stream): the open period is then closed on it, so the run still has
            a last bar. A known-end feed never completes a partial period --
            whether it holds one complete period or none must not change the
            rule the periods it does hold are closed by.
        :return: The completed HTF bar to run ``main()`` on, or ``None`` when
            this chart bar falls inside an open HTF period.
        """
        period_start = self.period_start(candle.timestamp)
        period_close = self._period_close(period_start)
        if self._chart_span_ms and candle.timestamp + self._chart_span_ms < period_close:
            # Cheap reject: a chart bar's real close never exceeds its nominal
            # one, so this bar cannot possibly reach the period close -- and a
            # bar that cannot reach it cannot straddle it either.
            complete = False
        else:
            chart_close = self._chart_close(candle.timestamp)
            complete = chart_close >= period_close
            if complete and self.period_start(chart_close - 1) != period_start:
                # The constructor's nesting check works on timeframe strings and
                # so can only prove that the spans divide; the feed's actual
                # PHASE is only known here. A chart grid offset from the script's
                # boundaries (UTC-aligned 120-minute bars under a daily script on
                # a UTC+1 exchange: the 22:00 UTC bar runs 23:00-01:00 local and
                # crosses local midnight) divides perfectly yet still straddles.
                raise ValueError(
                    f"Chart bar at {candle.timestamp} on timeframe "
                    f"'{self._chart_timeframe}' crosses a '{self.timeframe}' "
                    f"period boundary: the chart grid is offset from the "
                    f"script's periods, so its bars cannot be folded into them "
                    f"without corrupting both. Load the data on a timeframe "
                    f"whose bars align with the script's period boundaries."
                )
        forced = last_chart_bar and not self._feed_end_known
        _, _, closed = self._aggregator.update(
            candle.timestamp, candle.open, candle.high, candle.low,
            candle.close, candle.volume, period_complete=complete or forced)
        if closed is None:
            return None
        self._is_last = forced or (self._last_period is not None
                                   and closed.period_start >= self._last_period)
        return OHLCV(closed.period_start, closed.open, closed.high, closed.low,
                     closed.close, closed.volume, candle.extra_fields)

    @property
    def is_last(self) -> bool:
        """True when the bar last returned by :meth:`feed` is the script's last."""
        return self._is_last

    def next_period_start(self, next_time_ms: int | None) -> int:
        """
        Open time of the HTF bar that follows the one just executed.

        Mirrors the chart loop's one-bar peek: ``lib._next_time`` must name the
        next bar the SCRIPT will see, which security confirmation reads.

        :param next_time_ms: Next chart bar's open time, or ``None`` at the end
            of the feed.
        :return: The next HTF period's open time, or ``0`` when there is none.
        """
        if next_time_ms is None:
            return 0
        return self.period_start(next_time_ms)

    def remember(self, values: dict[str, float]) -> None:
        """
        Keep the plot values of the HTF bar just executed.

        :param values: ``lib._plot_data`` of the finished execution.
        """
        self._values = dict(values)
        if self._na_values is None or list(self._na_values) != list(values):
            self._na_values = {key: na_float for key in values}

    @property
    def gap_values(self) -> dict[str, float] | None:
        """
        Plot values for a chart bar inside an open HTF period.

        ``None`` before the first execution: no plot column is known yet, so
        there is no row to write.
        """
        if self._values is None:
            return None
        return dict(self._values if not self.gaps else self._na_values or {})
