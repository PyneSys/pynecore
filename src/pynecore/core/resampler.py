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


# Three midnight-crossing predicates on a session interval's (start, end)
# times-of-day. They are deliberately distinct rules — kept here as the single
# definition each so the same test is never re-spelled slightly differently
# across modules (the historical source of trading-day grouping bugs):
#
#   crosses_midnight   end <= start            the interval ends on the next
#                                              calendar day (close cap, security)
#   rolls_trading_day  crosses & not midnight  the open belongs to the next
#                                              trading day (overnight roll)
#
# The strict ``end < start`` session-overlap test (``lib.timeframe`` /
# ``lib.session``) is a third, genuinely different rule and stays inline there.


def crosses_midnight(start: dt_time, end: dt_time) -> bool:
    """
    Whether a session interval ends on the next calendar day.

    :param start: Interval open time-of-day
    :param end: Interval close time-of-day
    :return: ``True`` when the close is at or before the open
    """
    return end <= start


def rolls_trading_day(start: dt_time, end: dt_time) -> bool:
    """
    Whether an overnight session open rolls the trading day forward.

    An interval rolls the trading day when it crosses midnight but does not open
    exactly at midnight (a ``00:00`` open belongs to its own calendar day).

    :param start: Interval open time-of-day
    :param end: Interval close time-of-day
    :return: ``True`` for evening opens of the following trading day
    """
    return crosses_midnight(start, end) and not (
        start.hour == 0 and start.minute == 0 and start.second == 0)


def overnight_opens(
        opening_hours: 'list[SymInfoInterval] | None',
        session_starts: 'list[SymInfoSession] | None' = None) -> dict[int, dt_time]:
    """
    Earliest trading-day-rolling session open per open-weekday.

    Only sessions that roll the trading day count (:func:`rolls_trading_day`) —
    same rule as ``lib.time_tradingday``. Without ``opening_hours`` the overnight
    status is inferred from ``session_starts`` alone: an open at or after 12:00
    is an evening open of the next trading day (true for every overnight market
    we know — CME 17:00, FX 17:00 — while day sessions open in the morning).

    :param opening_hours: ``SymInfo.opening_hours`` (preferred source)
    :param session_starts: ``SymInfo.session_starts`` fallback
    :return: weekday -> earliest rolling open time
    """
    res: dict[int, dt_time] = {}
    if opening_hours:
        for day, start, end in opening_hours:
            if rolls_trading_day(start, end):
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


def overnight_starts_by_weekday(
        opening_hours: 'list[SymInfoInterval] | None') -> dict[int, list[dt_time]]:
    """
    All trading-day-rolling session opens per open-weekday (list shape).

    Same selection as :func:`overnight_opens` (:func:`rolls_trading_day`) but
    keeps every rolling open, not just the earliest — the per-bar overnight-roll
    code in ``lib.time_tradingday`` / ``lib.time_close`` walks the full list.

    :param opening_hours: ``SymInfo.opening_hours``
    :return: weekday -> list of rolling open times
    """
    res: dict[int, list[dt_time]] = {}
    if opening_hours:
        for day, start, end in opening_hours:
            if rolls_trading_day(start, end):
                res.setdefault(day, []).append(start)
    return res


def close_table_by_weekday(
        opening_hours: 'list[SymInfoInterval]',
        overnight_by_wd: dict[int, list[dt_time]]) -> dict[int, tuple[dt_time, int]]:
    """
    Per-trading-day-weekday closing instant of the session that ends that day.

    Each interval's end is assigned to the trading day it closes — rolled to the
    next day when the end lies inside an overnight session — and the latest end
    per trading day wins (a lunch-break morning end loses to the afternoon
    close). The result maps a trading day's weekday to its close as a
    ``(time-of-day, calendar-day offset from the trading-day date)`` pair.

    :param opening_hours: ``SymInfo.opening_hours`` (``SymInfoInterval`` list)
    :param overnight_by_wd: Rolling opens from :func:`overnight_starts_by_weekday`
    :return: trading-day weekday -> (close time-of-day, +days offset)
    """
    midnight = dt_time(0, 0, 0)
    best: dict[int, tuple[int, dt_time, int]] = {}  # td_wd -> (sort key, tod, +days)
    for day, start, end in opening_hours:
        crosses = crosses_midnight(start, end)  # the interval ends next calendar day
        if end == midnight:
            # The instant just before the end is still on the opening day
            eps_day = day
            rolled = bool(overnight_by_wd.get(eps_day))
        else:
            eps_day = (day + 1) % 7 if crosses else day
            rolled = any(end > o for o in overnight_by_wd.get(eps_day, ()))
        td_wd = (eps_day + 1) % 7 if rolled else eps_day
        end_cal_day = (day + 1) % 7 if crosses else day
        offset = (end_cal_day - td_wd) % 7
        key = offset * 86_400 + end.hour * 3600 + end.minute * 60 + end.second
        prev = best.get(td_wd)
        if prev is None or key > prev[0]:
            best[td_wd] = (key, end, offset)
    return {wd: (tod, offset) for wd, (_key, tod, offset) in best.items()}


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
        return int(datetime(d.year, d.month, d.day, tzinfo=tz).timestamp())
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


def observed_week_key(d: date) -> tuple[int, int]:
    """
    Weekly grid coordinates of trading day ``d``.

    A week belongs to its Monday's calendar year and weeks count from that
    year's first Monday. The single definition of the nD/nW/nM weekly grouping
    used by the aggregator, the bar magnifier and the ``_dg_*`` tracker.

    :param d: Trading day date
    :return: ``(week's year, 0-based week ordinal within that year)``
    """
    monday = d - timedelta(days=d.weekday())
    return monday.year, (monday - first_monday(monday.year)).days // 7


class ObservedDayCounter:
    """
    Year-reset observed-trading-day counter for multi-period grouping.

    Feed consecutive trading days; :meth:`ordinal` returns each day's in-year
    scheduled ordinal and :meth:`key` turns it into the nD/nW/nM grouping key.
    'Observed' symbols realize TradingView's holiday calendar through their
    actual daily data, so counting the days present in the stream reproduces the
    grid. The first day seeds the counter from the weekday grid — the days
    between Jan 1 and the stream start are not observable (exact when the stream
    begins at the year's first session).

    Holiday half-day fold (intraday source only): TradingView's daily feed does
    not emit a separate bar for the holiday half-day adjacent to an early close
    (e.g. the day after a 13:00 Thanksgiving close) — it folds into the
    early-close day. Detected from data alone, no calendar: a trading day is an
    early close when its last real bar ends before its scheduled session close
    (:meth:`_is_early`); the immediately following day then shares its ordinal
    instead of advancing. A folded day is itself not a trigger (chain-stop). Fold
    needs the per-bar end instants — pass ``bar_end`` to :meth:`ordinal`, or feed
    the previous day's last bar end via :meth:`note_bar_end` before rolling.

    A bare ``ObservedDayCounter()`` (no template, ``fold=False``) is the plain
    year-reset counter — ``ordinal(td)`` advances one slot per present day.
    """

    __slots__ = ('_cur', '_ordinal', '_cur_end', '_folded', '_fold', '_tz',
                 '_close_table')

    def __init__(self,
                 tz: 'ZoneInfo | dt_timezone | None' = None,
                 opening_hours: 'list[SymInfoInterval] | None' = None,
                 fold: bool = False) -> None:
        """
        :param tz: Exchange timezone (only needed when ``fold`` is on)
        :param opening_hours: ``SymInfo.opening_hours`` for the fold's scheduled
            close table (no template -> fold disabled)
        :param fold: Enable holiday half-day folding (intraday source only)
        """
        self._cur: date | None = None
        self._ordinal = 0
        self._cur_end: int | None = None  # running max bar-end of the current day
        self._folded = False  # current day folded into the previous (chain-stop)
        self._fold = fold and bool(opening_hours)
        self._tz = tz
        if self._fold:
            assert opening_hours is not None
            self._close_table = close_table_by_weekday(
                opening_hours, overnight_starts_by_weekday(opening_hours))
        else:
            self._close_table: dict[int, tuple[dt_time, int]] = {}

    def note_bar_end(self, bar_end: int | None) -> None:
        """
        Record a bar's end instant (epoch seconds) for the current trading day.

        Keeps the running maximum so the fold can tell whether the day closed
        early. Used when the counter does not see every bar (the ``_dg_*``
        tracker fires once per trading day): feed the previous day's last bar end
        before calling :meth:`ordinal` for the new day.

        :param bar_end: Bar end instant in epoch seconds, or ``None`` (no-op)
        """
        if bar_end is not None and (self._cur_end is None or bar_end > self._cur_end):
            self._cur_end = bar_end

    def _is_early(self, day: date, end: int | None) -> bool:
        """
        Whether ``day``'s last real bar ``end`` precedes its scheduled close.

        :param day: Trading day
        :param end: The day's last bar end in epoch seconds
        :return: ``True`` for an early-close (holiday half) day
        """
        if end is None:
            return False
        entry = self._close_table.get(day.weekday())
        if entry is None:
            return False
        end_tod, offset = entry
        close_date = day + timedelta(days=offset)
        close_sec = int(datetime(
            close_date.year, close_date.month, close_date.day,
            end_tod.hour, end_tod.minute, end_tod.second,
            tzinfo=self._tz).timestamp())
        return end < close_sec

    def ordinal(self, td: date, bar_end: int | None = None) -> int:
        """
        In-year scheduled ordinal of trading day ``td``.

        :param td: Trading day of the current bar (must not decrease)
        :param bar_end: This bar's end instant in epoch seconds (fold only)
        :return: 0-based ordinal within ``td``'s year
        """
        cur = self._cur
        if td != cur:
            if cur is None:
                self._ordinal = weekday_ordinal(td)
                self._folded = False
            elif td.year != cur.year:
                self._ordinal = 0
                self._folded = False
            elif (self._fold and not self._folded
                  and (td - cur).days == 1
                  and self._is_early(cur, self._cur_end)):
                # td is the holiday half folding into the early-close day cur:
                # share its ordinal, and do not let the next day fold too.
                self._folded = True
            else:
                self._ordinal += 1
                self._folded = False
            self._cur = td
            self._cur_end = bar_end
        elif bar_end is not None and (self._cur_end is None or bar_end > self._cur_end):
            self._cur_end = bar_end
        return self._ordinal

    def key(self, modifier: str, multiplier: int) -> tuple[int, int]:
        """
        nD/nW/nM grouping key of the current trading day.

        :param modifier: 'D', 'W' or 'M'
        :param multiplier: Period multiplier (> 1)
        :return: Period identity key
        """
        cur = self._cur
        assert cur is not None
        if modifier == 'D':
            return cur.year, self._ordinal // multiplier
        if modifier == 'W':
            wy, week = observed_week_key(cur)
            return wy, week // multiplier
        return cur.year, (cur.month - 1) // multiplier


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
