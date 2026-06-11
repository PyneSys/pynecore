from datetime import datetime, timedelta, timezone as dt_timezone, date, time as dt_time
from typing import ClassVar, TYPE_CHECKING
from functools import lru_cache
from zoneinfo import ZoneInfo

from ..lib import timeframe as tf_module

if TYPE_CHECKING:
    from .syminfo import SymInfoSession, SymInfoInterval


def _session_anchor_sec(
        t_sec: int,
        tz: ZoneInfo | dt_timezone | None,
        session_starts: 'list[SymInfoSession]',
) -> int:
    """
    Epoch seconds of the session open anchoring ``t_sec``'s trading session.

    Returns the opening instant of the trading session that *contains* ``t_sec`` —
    for overnight sessions this is the previous evening's open. Used to align
    intraday HTF bars to the session open the way TradingView does, instead of
    flooring to the UTC clock.

    Sessions are assumed shorter than 24h, so the containing open lies on the bar's
    local date or the one before it. Falls back to ``0`` — which makes the caller
    reproduce the plain UTC clock-floor — when no declared session covers ``t_sec``.

    :param t_sec: Bar timestamp in epoch seconds.
    :param tz: Exchange timezone for locating session opens. ``None`` uses local time.
    :param session_starts: Per-trading-day primary opens (``SymInfoSession``).
    :return: Anchor (session-open) time in epoch seconds, or 0 if none applies.
    """
    if tz is not None:
        local_date = datetime.fromtimestamp(t_sec, tz=tz).date()
    else:
        local_date = datetime.fromtimestamp(t_sec).date()

    best: int | None = None
    for delta in (0, 1):
        d = local_date - timedelta(days=delta)
        weekday = d.weekday()
        for s in session_starts:
            if s.day != weekday:
                continue
            open_sec = int(datetime(
                d.year, d.month, d.day,
                s.time.hour, s.time.minute, s.time.second,
                tzinfo=tz,
            ).timestamp())
            if open_sec <= t_sec and (best is None or open_sec > best):
                best = open_sec
    return best if best is not None else 0


#
# Multi-period (nD/nW/nM) scheduled grid — TradingView counts *scheduled trading
# days* on a per-exchange calendar, restarts the counter at each year's first
# scheduled day, and stamps every period with its first scheduled day's session
# open (synthetic when that day has no data). Which days are scheduled depends
# on the market:
#
#   'calendar' — 24/7 markets (crypto): every calendar day. Pure arithmetic,
#                exact for any data window.
#   'weekday'  — FX/CFD feeds: every Mon-Fri weekday. TradingView has no FX
#                holiday calendar, so Dec 25 / Jan 1 consume a grid slot even
#                with no data. Pure arithmetic, exact for any data window.
#   'observed' — exchange-listed symbols (futures, stocks): TradingView's real
#                holiday calendar, which we don't have — but the daily data is
#                its realization, so counting actual trading days reproduces it
#                (verified 100% on CME 2022+). Needs the data stream; the
#                arithmetic here serves only as the dataless fallback.
#
# Verified against TradingView: OANDA/CAPITALCOM EURUSD 5D 2002-2026 (weekday,
# 100%), BINANCE:BTCUSDT 5D 2017-2026 (calendar, 100%), CME_MINI:RTY1! 5D/2D/6D
# 2022-2026 (observed, 100%) and RTY 3W/3M (week/month grids).
#
# On intraday charts a chart bar belongs to the period its *last* instant falls
# into: the bar containing a session open is the new trading day's first bar
# even when its own timestamp precedes the open (CAPITALCOM EURUSD opens
# Mon-Thu at 17:05 ET — on a 240-minute grid the 17:00 bar starts the new day).
# The functions below map plain instants to the grid; callers resolve chart
# bars by passing ``bar open + chart span - 1``. D/W/M chart bars are
# session-aligned by construction and need no offset.
#


def grid_mode(sym_type: str | None,
              opening_hours: 'list[SymInfoInterval] | None') -> str:
    """
    Classify the symbol's scheduled-trading-day calendar.

    :param sym_type: ``SymInfo.type`` (e.g. "forex", "futures", "crypto")
    :param opening_hours: ``SymInfo.opening_hours`` intervals
    :return: ``'calendar'``, ``'weekday'`` or ``'observed'``
    """
    if opening_hours:
        days = {day for day, _start, _end in opening_hours}
        if len(days) == 7:
            return 'calendar'
    if sym_type in ('crypto', 'spot', 'swap'):
        return 'calendar'
    if sym_type == 'forex':
        return 'weekday'
    return 'observed'


def overnight_opens(
        opening_hours: 'list[SymInfoInterval] | None',
        session_starts: 'list[SymInfoSession] | None' = None) -> dict[int, dt_time]:
    """
    Earliest trading-day-rolling session open per open-weekday.

    Only sessions that cross midnight roll the trading day (end at or before
    their start, not opening exactly at midnight) — same rule as
    ``lib.time_tradingday``. Without ``opening_hours`` the overnight status is
    inferred from ``session_starts`` alone: an open at or after 12:00 is an
    evening open of the next trading day (true for every overnight market we
    know — CME 17:00, FX 17:00 — while day sessions open in the morning).

    :param opening_hours: ``SymInfo.opening_hours`` (preferred source)
    :param session_starts: ``SymInfo.session_starts`` fallback
    :return: weekday -> earliest rolling open time
    """
    res: dict[int, dt_time] = {}
    if opening_hours:
        for day, start, end in opening_hours:
            if end <= start and not (start.hour == 0 and start.minute == 0
                                     and start.second == 0):
                cur = res.get(day)
                if cur is None or start < cur:
                    res[day] = start
    elif session_starts:
        for day, start in session_starts:
            if start.hour >= 12:
                cur = res.get(day)
                if cur is None or start < cur:
                    res[day] = start
    return res


def trading_day(ts_sec: float, tz: ZoneInfo | dt_timezone | None,
                overnight: dict[int, dt_time]) -> date:
    """
    Calendar date of the trading day a timestamp belongs to.

    A bar at or after its weekday's overnight session open belongs to the next
    calendar day (CME Sunday 17:00 open -> Monday's trading day).

    :param ts_sec: Timestamp in epoch seconds
    :param tz: Exchange timezone (``None`` uses the system's local timezone)
    :param overnight: Per-weekday rolling opens from :func:`overnight_opens`
    :return: The trading day's calendar date
    """
    dt_loc = datetime.fromtimestamp(ts_sec, tz)
    d = dt_loc.date()
    if overnight:
        t0 = overnight.get(dt_loc.weekday())
        if t0 is not None and dt_loc.time() >= t0:
            d += timedelta(days=1)
    return d


def weekday_ordinal(d: date) -> int:
    """
    Index of ``d`` among its year's Mon-Fri weekdays (Jan 1 weekday = 0).

    :param d: A weekday date
    :return: 0-based scheduled-day ordinal in the 'weekday' grid
    """
    days = (d - date(d.year, 1, 1)).days
    weeks, rem = divmod(days, 7)
    n = weeks * 5
    w0 = date(d.year, 1, 1).weekday()
    for i in range(rem):
        if (w0 + i) % 7 < 5:
            n += 1
    return n


def weekday_from_ordinal(year: int, idx: int) -> date:
    """
    Inverse of :func:`weekday_ordinal`: the year's ``idx``-th Mon-Fri weekday.

    :param year: Calendar year
    :param idx: 0-based weekday ordinal
    :return: The weekday's date
    """
    jan1 = date(year, 1, 1)
    w0 = jan1.weekday()
    base = jan1 + timedelta(days=0 if w0 < 5 else 7 - w0)
    weeks, rem = divmod(idx, 5)
    base += timedelta(weeks=weeks)
    bw = base.weekday()
    return base + timedelta(days=rem if bw + rem <= 4 else rem + 2)


def trading_day_open_sec(d: date, tz: ZoneInfo | dt_timezone | None,
                         session_starts: 'list[SymInfoSession] | None',
                         overnight: dict[int, dt_time]) -> int:
    """
    Epoch seconds of the session open that begins trading day ``d``.

    For overnight markets this is the previous day's evening open (CME Monday
    trading day -> Sunday 17:00). The instant exists on the schedule even when
    the exchange was closed that day — TradingView stamps multi-period bars
    with these synthetic opens (e.g. FX 5D bars opening on a dataless Jan 1).
    Falls back to ``d``'s local midnight when no template entry matches.

    :param d: Trading day date
    :param tz: Exchange timezone (``None`` uses the system's local timezone)
    :param session_starts: ``SymInfo.session_starts`` template
    :param overnight: Per-weekday rolling opens from :func:`overnight_opens`
    :return: Session open in epoch seconds
    """
    best: int | None = None
    if session_starts:
        for delta in (1, 0):
            od = d - timedelta(days=delta)
            w = od.weekday()
            for day, t in session_starts:
                if day != w:
                    continue
                rolls = w in overnight and t >= overnight[w]
                if (delta == 1) != rolls:
                    continue
                sec = int(datetime(od.year, od.month, od.day,
                                   t.hour, t.minute, t.second, tzinfo=tz).timestamp())
                if best is None or sec < best:
                    best = sec
    if best is None:
        best = int(datetime(d.year, d.month, d.day, tzinfo=tz).timestamp())
    return best


def scheduled_day_ordinal(d: date, mode: str) -> int:
    """
    0-based ordinal of trading day ``d`` on its year's scheduled-day grid.

    :param d: Trading day date
    :param mode: ``'calendar'`` or ``'weekday'`` (``'observed'`` has no
                 dataless ordinal — callers count the data stream instead and
                 use the weekday grid only as pre-window approximation)
    :return: Scheduled-day ordinal within the year
    """
    if mode == 'calendar':
        return (d - date(d.year, 1, 1)).days
    return weekday_ordinal(d)


def scheduled_day_from_ordinal(year: int, idx: int, mode: str) -> date:
    """
    Inverse of :func:`scheduled_day_ordinal`.

    :param year: Calendar year
    :param idx: 0-based scheduled-day ordinal
    :param mode: ``'calendar'`` or ``'weekday'``
    :return: The scheduled day's date
    """
    if mode == 'calendar':
        return date(year, 1, 1) + timedelta(days=idx)
    return weekday_from_ordinal(year, idx)


def first_monday(year: int) -> date:
    """
    The year's first Monday — anchor of the weekly grid.

    A week belongs to its Monday's calendar year, so the week containing
    Jan 1 belongs to the previous year unless Jan 1 is a Monday.

    :param year: Calendar year
    :return: Date of the first Monday
    """
    jan1 = date(year, 1, 1)
    w0 = jan1.weekday()
    return jan1 + timedelta(days=0 if w0 == 0 else 7 - w0)


class ObservedDayCounter:
    """
    Year-reset observed-trading-day counter for multi-period grouping.

    Feed consecutive trading days; :meth:`ordinal` returns each day's in-year
    scheduled ordinal. 'Observed' symbols realize TradingView's holiday
    calendar through their actual daily data, so counting the days present in
    the stream reproduces the grid. The first day seeds the counter from the
    weekday grid — the days between Jan 1 and the stream start are not
    observable (exact when the stream begins at the year's first session).
    """

    __slots__ = ('_cur', '_ordinal')

    def __init__(self) -> None:
        self._cur: date | None = None
        self._ordinal = 0

    def ordinal(self, td: date) -> int:
        """
        In-year scheduled ordinal of trading day ``td``.

        :param td: Trading day of the current bar (must not decrease)
        :return: 0-based ordinal within ``td``'s year
        """
        if td != self._cur:
            if self._cur is None:
                self._ordinal = weekday_ordinal(td)
            elif td.year != self._cur.year:
                self._ordinal = 0
            else:
                self._ordinal += 1
            self._cur = td
        return self._ordinal


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
                     tz: ZoneInfo | dt_timezone | None = None,
                     session_starts: 'list[SymInfoSession] | None' = None,
                     opening_hours: 'list[SymInfoInterval] | None' = None,
                     mode: str | None = None) -> int:
        """
        Calculate the bar opening time for the current timeframe.

        For daily, weekly, and monthly timeframes, the timezone determines where
        midnight falls — i.e., which calendar day a timestamp belongs to.

        For intraday timeframes the bar grid is, by default, a pure UTC-epoch
        clock-floor (session-unaware fast path). When ``session_starts`` is given
        the grid is instead anchored to the session open — matching TradingView,
        which aligns intraday HTF bars to the session open rather than the UTC
        clock (e.g. a 09:30 open at 1H yields 09:30, 10:30, 11:30…). For sessions
        that open on a ``tf`` boundary the two are identical; callers therefore
        pass ``session_starts`` only when the open is actually off-grid, so the
        common case keeps the zero-overhead fast path.

        Multi-period timeframes (nD/nW/nM with n > 1) live on the year-reset
        scheduled grid (see the module docs): periods count scheduled trading
        days/weeks/months from the year's first scheduled one, and each period
        is stamped with its first scheduled day's session open. For 'observed'
        symbols (exchange-listed) this arithmetic is only the dataless fallback
        on the weekday grid — data-driven callers count actual trading days.

        :param current_time_ms: Current time in milliseconds (UNIX timestamp)
        :param tz: Timezone for day/week/month boundary calculation, and for
                   locating session opens when ``session_starts`` is given.
                   If None, uses the system's local timezone.
        :param session_starts: Per-trading-day primary opens for intraday session
                   anchoring and for multi-period session-open stamps.
                   ``None`` (default) selects the pure clock-floor.
        :param opening_hours: ``SymInfo.opening_hours`` — preferred source for
                   the trading-day roll of multi-period grids (inferred from
                   ``session_starts`` when omitted).
        :param mode: Scheduled-grid mode from :func:`grid_mode`; ``None`` infers
                   'calendar' when the session template covers all 7 days,
                   'weekday' otherwise. Only multi-period timeframes use it.
        :return: Bar opening time in milliseconds
        """
        # Convert to seconds for calculations
        current_time_sec = current_time_ms // 1000

        # Get timeframe in seconds
        tf_seconds = tf_module.in_seconds(self.timeframe)

        # Calculate bar opening time based on timeframe type
        # noinspection PyProtectedMember
        modifier, multiplier = tf_module._process_tf(self.timeframe)

        if modifier in ('S', ''):  # Seconds / minutes (intraday)
            if session_starts is None:
                # Pure UTC-epoch clock-floor — session-unaware fast path
                bar_start_sec = (current_time_sec // tf_seconds) * tf_seconds
            else:
                # Session-anchored grid. Anchor and step in absolute epoch
                # seconds so the boundaries stay correct across DST transitions.
                anchor_sec = _session_anchor_sec(current_time_sec, tz, session_starts)
                bar_start_sec = anchor_sec + (
                    (current_time_sec - anchor_sec) // tf_seconds) * tf_seconds

        elif modifier in ('D', 'W', 'M') and multiplier > 1:
            # Multi-period: year-reset scheduled grid stamped at session opens
            on = overnight_opens(opening_hours, session_starts)
            if mode in ('calendar', 'weekday'):
                gmode = mode
            else:
                day_set = {day for day, _t in session_starts} if session_starts else set()
                gmode = 'calendar' if len(day_set) == 7 else 'weekday'
            td = trading_day(current_time_sec, tz, on)

            if modifier == 'D':
                idx = scheduled_day_ordinal(td, gmode)
                slot = scheduled_day_from_ordinal(
                    td.year, (idx // multiplier) * multiplier, gmode)

            elif modifier == 'W':
                # A week belongs to its Monday's year; weeks count from the
                # year's first Monday.
                monday = td - timedelta(days=td.weekday())
                fm = first_monday(monday.year)
                weeks = (monday - fm).days // 7
                slot = fm + timedelta(weeks=(weeks // multiplier) * multiplier)

            else:  # 'M' — calendar months grouped within the year
                m0 = ((td.month - 1) // multiplier) * multiplier + 1
                slot = date(td.year, m0, 1)
                if gmode == 'weekday' and slot.weekday() >= 5:
                    # First scheduled day of the period's first month
                    slot += timedelta(days=7 - slot.weekday())

            bar_start_sec = trading_day_open_sec(slot, tz, session_starts, on)

        elif modifier in ('D', 'W', 'M'):
            # Daily/Weekly/Monthly — timezone matters for calendar alignment
            if tz is not None:
                current_dt = datetime.fromtimestamp(current_time_sec, tz=tz)
            else:
                current_dt = datetime.fromtimestamp(current_time_sec)

            if modifier == 'D':  # Daily
                bar_start_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)

            elif modifier == 'W':  # Weekly
                bar_start_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                days_to_monday = bar_start_dt.weekday()  # 0 = Monday
                bar_start_dt -= timedelta(days=days_to_monday)

            else:  # Monthly
                bar_start_dt = current_dt.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0)

            bar_start_sec = int(bar_start_dt.timestamp())

        else:
            raise ValueError(f"Unsupported timeframe modifier: {modifier}")

        # Convert back to milliseconds
        return bar_start_sec * 1000
