from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.types.type_checker import *

from functools import lru_cache
from datetime import datetime, timedelta

from ..core.module_property import module_property

from .. import lib
from . import syminfo as _syminfo

from ._timeframe_change import change

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
        tf_sec: int = in_seconds(_syminfo.period)
    if prev_dt is None:
        prev_dt: datetime = current_dt - timedelta(seconds=tf_sec)

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


def in_seconds(timeframe: str | None = None) -> int:
    """
    Convert the timeframe to seconds

    :param timeframe: The timeframe to convert. If not provided or an empty string,
        uses the current chart timeframe (Pine treats ``""`` as the chart timeframe).
    :return: The timeframe in seconds
    :raises ValueError: If the timeframe is invalid
    """
    if not timeframe:
        timeframe: str = str(_syminfo.period)
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
    if lib._script is None:
        return str(_syminfo.period)
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
