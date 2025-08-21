from functools import lru_cache
from datetime import datetime, timedelta, UTC

from ..core.module_property import module_property

from .. import lib
from . import syminfo as _syminfo
from pynecore.core.datetime import parse_timezone as _parse_timezone

__all__ = [
    'change',
    'from_seconds',
    'in_seconds',
    'isdaily',
    'isdwm',
    'isintraday',
    'isminutes',
    'ismonthly',
    'isseconds',
    'isticks',
    'isweekly',
    'main_period',
    'multiplier',
    'period'
]

__persistent_function_vars__ = {}


@lru_cache(maxsize=128)
def _process_tf(timeframe: str) -> tuple[str, int]:
    """
    Process the timeframe string and return the modi    fier and multiplier

    :param timeframe: The timeframe string in TradingView format
    :return: A tuple with the modifier and multiplier
    :raises AssertionError: If the timeframe is invalid
    """
    assert len(timeframe) > 0, "Invalid timeframe: empty string!"

    # Simple minutes
    if timeframe.isdigit():
        _modifier = ''
        _multiplier = int(timeframe)

    # Multiplier and modifier
    elif len(timeframe) > 1:
        if not timeframe[-1].isdigit():
            _modifier = timeframe[-1]
            _multiplier = int(timeframe[:-1])
        else:
            raise AssertionError("Invalid timeframe format!")

    # Just a single character
    else:
        _modifier = timeframe
        _multiplier = 1

    assert _modifier in ('', 'T', 'S', 'D', 'W', 'M'), "Invalid timeframe: wrong modifier!"
    assert _multiplier > 0, "Invalid timeframe: wrong multiplier!"
    return _modifier, _multiplier


def _is_time_in_candle(candle_time: datetime, check_time: datetime, tf_sec: int) -> bool:
    """
    Check if a given time is within the candle's timeframe.

    :param candle_time: Start datetime of the candle
    :param check_time: Time to check
    :param tf_sec: Timeframe in seconds
    :return: True if the time is within the candle
    """
    candle_end = candle_time + timedelta(seconds=tf_sec)
    return candle_time <= check_time < candle_end


# noinspection PyProtectedMember
def _get_first_session_start_in_day(dt: datetime) -> datetime | None:
    """
    Get first session start time in the given day using session_starts configuration.

    :param dt: The datetime to check (in UTC)
    :return: First session start time in UTC
    """
    local_dt = dt.astimezone(_parse_timezone(_syminfo.timezone))
    weekday = local_dt.weekday()

    # Get all session starts for current weekday
    day_starts = [
        time for day, time in _syminfo._session_starts
        if day == weekday
    ]

    if not day_starts:
        return None

    # Get the earliest session start for the day
    first_start = min(day_starts)

    # Create datetime with the session start time
    ssdt = local_dt.replace(
        hour=first_start.hour,
        minute=first_start.minute,
        second=first_start.second,
        microsecond=0
    )

    return ssdt.astimezone(UTC)


# noinspection PyProtectedMember
def _get_new_year_session(dt: datetime) -> datetime | None:
    """
    Get the new year session of the given datetime.

    :param dt: The datetime to start searching in UTC
    :return: The next new year session in exchange timezone
    """
    dt_utc = dt.astimezone(UTC)
    nydt_utc = dt_utc.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    # Get next new year
    while True:
        ss = _get_first_session_start_in_day(nydt_utc)
        if ss is None:
            nydt_utc += timedelta(days=1)
            continue
        # Found the next new year session, which will be the next anchor
        return ss.astimezone(_parse_timezone(_syminfo.timezone))


# noinspection PyProtectedMember
def _is_new_session(current_dt: datetime, prev_dt: datetime | None = None, tf_sec: int | None = None) -> bool:
    """
    Check if current bar starts a new session.

    :param current_dt: Current candle datetime (in local exchange timezone)
    :param prev_dt: Previous candle datetime (in local exchange timezone)
    :param tf_sec: Timeframe width in seconds
    :return: True if this is the first candle of a new session
    """
    if tf_sec is None:
        tf_sec = in_seconds(_syminfo.period)
    if prev_dt is None:
        prev_dt = current_dt - timedelta(seconds=tf_sec)

    current_weekday = current_dt.weekday()
    prev_weekday = prev_dt.weekday()

    # Get all possible session starts:
    # 1. Current day sessions
    # 2. If previous day different, then previous day's overnight sessions
    session_starts = [
        ss for day, ss in _syminfo._session_starts
        if day == current_weekday
    ]

    if prev_weekday != current_weekday:
        prev_day_starts = [
            (ss, se) for day, ss, se in _syminfo._opening_hours
            if day == prev_weekday and se < ss  # Only overnight sessions
        ]
        # Add overnight session starts to check
        for ss, se in prev_day_starts:
            session_starts.append(ss)

    # For each session start
    for start_time in session_starts:
        session_start = current_dt.replace(
            hour=start_time.hour,
            minute=start_time.minute,
            second=start_time.second,
            microsecond=0
        )

        # If it crosses the day boundary, we set it to the previous day
        if prev_weekday != current_weekday and start_time > current_dt.time():
            session_start = session_start - timedelta(days=1)

        # If session starts in this bar
        if (current_dt <= session_start < current_dt + timedelta(seconds=tf_sec) and
                session_start > prev_dt):
            return True

    return False


__persistent_next_new_year_session: datetime | None = None
__persistent_cycle: int = 0
__persistent_last_dt: datetime | None = None
__persistent_last_signal: datetime | None = None
__persistent_next_anchor: datetime | None = None
__persistent_next_signal: datetime | None = None
__persistent_function_vars__['change'] = ['__persistent_next_new_year_session', '__persistent_cycle',
                                          '__persistent_last_dt', '__persistent_next_anchor',
                                          '__persistent_next_signal', '__persistent_last_signal']


# noinspection PyProtectedMember
# TODO: make it simpler and better if you could
# I know this function is awful. It was one of the hardest part of the whole Pine library.
# It may be simplified. The problem is that every timefram has different anchor points and slightly different rules. Or
# just not found the general rule for all timeframes.
def change(timeframe: str) -> bool:
    """
    Detects changes in the specified timeframe.

    :param timeframe: The timeframe to check
    :return: Returns true on the first bar of a `timeframe`, false otherwise.
    """
    global __persistent_next_new_year_session, __persistent_cycle, __persistent_last_dt, \
        __persistent_next_anchor, __persistent_next_signal, __persistent_last_signal

    tf_sec = in_seconds(timeframe)
    xchg_tf = _syminfo.period
    xchg_tf_sec = in_seconds(xchg_tf)

    # The timeframe to check nust be greater (or equal) than the current timeframe
    if tf_sec < xchg_tf_sec:
        return False

    _modifier, _multiplier = _process_tf(timeframe)
    assert _modifier != 'T', "Ticks are not (yet) supported!"
    is_intraday = _modifier == '' or _modifier == 'S'

    # Datetime in exchange timezone
    dt: datetime = lib._datetime

    # Initialize persistent variables
    if __persistent_last_dt is None:
        __persistent_last_dt = dt - timedelta(seconds=xchg_tf_sec)
        __persistent_last_signal = __persistent_last_dt

    # Find anchor point and replay sessions to the 1st bar
    if not is_intraday and __persistent_next_new_year_session is None:
        dt_utc = dt.astimezone(UTC)
        nydt_utc = dt_utc.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        nydt = _get_new_year_session(nydt_utc)

        assert nydt is not None

        __persistent_next_new_year_session = _get_new_year_session(dt_utc.replace(year=dt_utc.year + 1))
        __persistent_cycle = 0

        # Daily timeframe's anchor is the first session of the year
        if _modifier == 'D':
            _dt = nydt

            while _dt < dt:
                if _is_new_session(_dt, _dt - timedelta(seconds=xchg_tf_sec), xchg_tf_sec):
                    __persistent_cycle -= 1
                    if __persistent_cycle <= 0:
                        __persistent_last_signal = _dt
                        __persistent_cycle = _multiplier

                _dt += timedelta(seconds=xchg_tf_sec)

        # Weekly timeframe's anchor is the first Monday of the year
        elif _modifier == 'W':
            _dt_utc = nydt_utc

            # The anchor point must be Monday
            while _dt_utc < dt_utc:
                if _dt_utc.weekday() != 0:
                    _dt_utc += timedelta(days=1)
                    continue
                break

            # Find the next signal
            while _dt_utc < dt_utc:
                _dt_utc += timedelta(seconds=tf_sec)

            __persistent_next_anchor = _dt_utc
            __persistent_next_signal = None
            while __persistent_next_signal is None:
                __persistent_next_signal = _get_first_session_start_in_day(_dt_utc)
                _dt_utc += timedelta(days=1)
            __persistent_next_signal = __persistent_next_signal.astimezone(_parse_timezone(_syminfo.timezone))

        # Monthly timeframe's anchor is the first day of the month
        elif _modifier == 'M':
            if dt_utc.month == 12:
                next_month_dt = dt_utc.replace(year=dt_utc.year + 1, month=1, day=1, hour=0, minute=0, second=0)
            else:
                next_month_dt = dt_utc.replace(month=dt_utc.month + 1, day=1, hour=0, minute=0, second=0)

            __persistent_next_signal = None
            while __persistent_next_signal is None:
                __persistent_next_signal = _get_first_session_start_in_day(next_month_dt)
                next_month_dt += timedelta(days=1)
            __persistent_next_signal = __persistent_next_signal.astimezone(_parse_timezone(_syminfo.timezone))

            __persistent_cycle = dt_utc.month

    # We need to check every virtual candles, even if they are missing in the dataset
    while __persistent_last_dt < dt:
        prev_dt = __persistent_last_dt
        __persistent_last_dt += timedelta(seconds=xchg_tf_sec)

        if is_intraday:
            # The anchor point is the session start
            if _is_new_session(__persistent_last_dt, prev_dt, xchg_tf_sec):
                # We need to round the session start to the nearest hour
                __persistent_last_signal = __persistent_last_dt.replace(minute=0, second=0, microsecond=0)
            assert __persistent_last_signal is not None
            seconds_since_last_session = (__persistent_last_dt - __persistent_last_signal).total_seconds()

            if seconds_since_last_session % tf_sec == 0:
                __persistent_last_dt = dt
                return lib.bar_index > 0

        # We need to check only trading days
        elif _modifier == 'D':
            # Check if it is a new year
            assert __persistent_next_new_year_session is not None
            if (_is_time_in_candle(__persistent_last_dt, __persistent_next_new_year_session, xchg_tf_sec) or
                    __persistent_next_new_year_session < __persistent_last_dt):
                # Find the next new year session
                __persistent_next_new_year_session = _get_new_year_session(
                    __persistent_next_new_year_session + timedelta(days=367))
                __persistent_cycle = 0

            if _is_new_session(__persistent_last_dt, prev_dt, xchg_tf_sec):
                __persistent_cycle -= 1
                assert __persistent_last_signal is not None
                if __persistent_cycle <= 0:
                    __persistent_cycle = _multiplier
                    if __persistent_last_signal < __persistent_last_dt:
                        __persistent_last_signal = dt
                        return lib.bar_index > 0

        # We don't need to skip weekends here
        elif _modifier == 'W':
            assert __persistent_next_new_year_session is not None
            if (_is_time_in_candle(__persistent_last_dt, __persistent_next_new_year_session, xchg_tf_sec) or
                    __persistent_next_new_year_session < __persistent_last_dt):
                _dt_utc = ((__persistent_next_new_year_session + timedelta(days=1))
                           .astimezone(UTC).replace(day=1, month=1, hour=0, minute=0, second=0))
                __persistent_next_new_year_session = _get_new_year_session(_dt_utc.replace(year=_dt_utc.year + 1))

                # The anchor point must be Monday
                while True:
                    if _dt_utc.weekday() != 0:
                        _dt_utc += timedelta(days=1)
                        continue
                    break

                # 1st signal of the new year
                __persistent_next_signal = None
                while __persistent_next_signal is None:
                    __persistent_next_signal = _get_first_session_start_in_day(_dt_utc)
                    _dt_utc += timedelta(days=1)
                __persistent_next_anchor = __persistent_next_signal
                __persistent_next_signal = __persistent_next_signal.astimezone(_parse_timezone(_syminfo.timezone))

            assert __persistent_next_signal is not None
            if (_is_time_in_candle(__persistent_last_dt, __persistent_next_signal, xchg_tf_sec)
                    or __persistent_last_dt >= __persistent_next_signal):
                assert __persistent_next_anchor is not None
                # Increase the anchor date
                __persistent_next_anchor += timedelta(seconds=tf_sec)
                # Find the next signal which is usually Monday
                _dt_utc = __persistent_next_anchor
                __persistent_next_signal = None
                while __persistent_next_signal is None:
                    __persistent_next_signal = _get_first_session_start_in_day(_dt_utc)
                    _dt_utc += timedelta(days=1)
                __persistent_next_signal = __persistent_next_signal.astimezone(_parse_timezone(_syminfo.timezone))
                return lib.bar_index > 0

        elif _modifier == 'M':
            # Is it a new year candle?
            assert __persistent_next_new_year_session is not None
            if (_is_time_in_candle(__persistent_last_dt, __persistent_next_new_year_session, xchg_tf_sec) or
                    __persistent_next_new_year_session < __persistent_last_dt):
                _dt_utc = ((__persistent_next_new_year_session + timedelta(days=1))
                           .astimezone(UTC).replace(day=1, month=1, hour=0, minute=0, second=0))
                __persistent_next_new_year_session = _get_new_year_session(_dt_utc.replace(year=_dt_utc.year + 1))

                # 1st signal of the new year
                __persistent_next_signal = None
                while __persistent_next_signal is None:
                    __persistent_next_signal = _get_first_session_start_in_day(_dt_utc)
                    _dt_utc += timedelta(days=1)
                __persistent_next_signal = __persistent_next_signal.astimezone(_parse_timezone(_syminfo.timezone))

                __persistent_cycle = 0

            assert __persistent_next_signal is not None
            if (_is_time_in_candle(__persistent_last_dt, __persistent_next_signal, xchg_tf_sec)
                    or __persistent_last_dt >= __persistent_next_signal):
                _dt_utc = __persistent_next_signal.astimezone(UTC) + timedelta(days=1)
                if _dt_utc.month == 12:
                    next_month_dt = _dt_utc.replace(year=_dt_utc.year + 1, month=1, day=1, hour=0, minute=0, second=0)
                else:
                    next_month_dt = _dt_utc.replace(month=_dt_utc.month + 1, day=1, hour=0, minute=0, second=0)
                __persistent_next_signal = None
                while __persistent_next_signal is None:
                    __persistent_next_signal = _get_first_session_start_in_day(next_month_dt)
                    next_month_dt += timedelta(days=1)

                __persistent_next_signal = __persistent_next_signal.astimezone(_parse_timezone(_syminfo.timezone))

                m = __persistent_cycle
                __persistent_cycle += 1

                if m % _multiplier == 0:
                    return lib.bar_index > 0

    return False


@lru_cache(maxsize=128)
def from_seconds(seconds: int) -> str:
    """
    Convert seconds to a timeframe

    :param seconds: The seconds to convert
    :return: The timeframe in TradingView format
    """
    if seconds % (60 * 60 * 24 * 7 * 4) == 0:
        return f"{seconds // (60 * 60 * 24 * 7 * 4)}M"
    if seconds % (60 * 60 * 24 * 7) == 0:
        return f"{seconds // (60 * 60 * 24 * 7)}W"
    if seconds % (60 * 60 * 24) == 0:
        return f"{seconds // (60 * 60 * 24)}D"
    if seconds % 60 == 0:
        return f"{seconds // 60}"
    return f"{seconds}S"


def in_seconds(timeframe: str) -> int:
    """
    Convert the timeframe to seconds

    :param timeframe: The timeframe to convert
    :return: The timeframe in seconds
    :raises ValueError: If the timeframe is invalid
    """
    _modifier, _multiplier = _process_tf(timeframe)
    if _modifier == 'S':
        return _multiplier
    elif _modifier == 'D':
        return _multiplier * 60 * 60 * 24
    elif _modifier == 'W':
        return _multiplier * 60 * 60 * 24 * 7
    elif _modifier == 'M':
        return int(_multiplier * (60 * 60 * 24 * (365 / 12) + 3))  # Don't know why, but TV adds 3 secs here
    elif _modifier == '':
        return _multiplier * 60
    else:
        raise ValueError("Not supported timeframe!")


@module_property
def isdaily() -> bool:
    """
    Check if the current timeframe is daily

    :return: True if the current timeframe is daily
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == 'D'


@module_property
def isdwm() -> bool:
    """
    Check if the current timeframe is intraday, daily, weekly or monthly

    :return: True if the current timeframe is intraday, daily, weekly or monthly
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == 'D' or modifier == 'W' or modifier == 'M'


@module_property
def isintraday() -> bool:
    """
    Check if the current timeframe is intraday

    :return: True if the current timeframe is intraday
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == '' or modifier == 'S' or modifier == 'T'


@module_property
def isminutes() -> bool:
    """
    Check if the current timeframe is minutes

    :return: True if the current timeframe is minutes
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == ''


@module_property
def ismonthly() -> bool:
    """
    Check if the current timeframe is monthly

    :return: True if the current timeframe is monthly
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == 'M'


@module_property
def isseconds() -> bool:
    """
    Check if the current timeframe is seconds

    :return: True if the current timeframe is seconds
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == 'S'


@module_property
def isticks() -> bool:
    """
    Check if the current timeframe is ticks

    :return: True if the current timeframe is ticks
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == 'T'


@module_property
def isweekly() -> bool:
    """
    Check if the current timeframe is weekly

    :return: True if the current timeframe is weekly
    """
    modifier, _ = _process_tf(_syminfo.period)
    return modifier == 'W'


# noinspection PyProtectedMember
@module_property
def main_period() -> str:
    """
    Get the main period

    :return: The main period
    """
    assert lib._script is not None
    return lib._script.timeframe or str(_syminfo.period)


@module_property
def multiplier() -> int:
    """
    Get the current timeframe multiplier

    :return: The current timeframe multiplier
    """
    _, _multiplier = _process_tf(_syminfo.period)
    return _multiplier


@module_property
def period() -> str:
    """
    Get the current period

    :return: The current period
    """
    return str(_syminfo.period)
