from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.types.type_checker import *

from functools import lru_cache

from ..errorcodes import (InvalidTimeframeError as _InvalidTimeframeError,
                     TickTimeframeError as _TickTimeframeError)
from ..core.module_property import module_property
from ..types.pine_types import PyneInt, pine_int

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
    :raises InvalidTimeframeError: If the timeframe is invalid
    """
    if len(timeframe) == 0:
        raise _InvalidTimeframeError(timeframe=timeframe)

    # Simple minutes
    if timeframe.isdigit():
        _modifier = ''
        _multiplier = int(timeframe)

    # Multiplier and modifier
    elif len(timeframe) > 1:
        if timeframe[-1].isdigit():
            raise _InvalidTimeframeError(timeframe=timeframe)
        _modifier = timeframe[-1]
        try:
            _multiplier = int(timeframe[:-1])
        except ValueError:
            raise _InvalidTimeframeError(timeframe=timeframe) from None

    # Just a single character
    else:
        _modifier = timeframe
        _multiplier = 1

    if _modifier not in ('', 'T', 'S', 'D', 'W', 'M') or not _multiplier > 0:
        raise _InvalidTimeframeError(timeframe=timeframe)
    return _modifier, _multiplier


# noinspection PyProtectedMember
def _current_period() -> str:
    """
    Timeframe the running script executes on.

    A script declared with ``indicator(..., timeframe='W')`` is evaluated on
    higher-timeframe bars, and every ``timeframe.*`` builtin describes THAT
    timeframe -- while ``syminfo.period`` keeps the chart (data feed) timeframe
    the bar grid is built from.

    :return: The script timeframe if the declaration carries one, else the chart's.
    """
    # A ``request.security()`` child is the one place that must answer with its
    # OWN period: it evaluates the expression in the context's timeframe, not the
    # script's. ``lib._main_timeframe`` is set only in such a child.
    if lib._main_timeframe is None and lib._script_timeframe:
        return lib._script_timeframe
    return str(_syminfo.period)


@lru_cache(maxsize=128)
def from_seconds(seconds: int) -> str:
    """
    Convert seconds to a timeframe

    :param seconds: The seconds to convert
    :return: The timeframe string (e.g. ``"60"``, ``"1D"``)
    """
    # A Pine int is a double at runtime, and the string this builds is a
    # timeframe: it is a consuming slot, so the count is truncated here
    seconds = int(seconds)
    if seconds % (60 * 60 * 24 * 7 * 4) == 0:
        return f"{seconds // (60 * 60 * 24 * 7 * 4)}M"
    if seconds % (60 * 60 * 24 * 7) == 0:
        return f"{seconds // (60 * 60 * 24 * 7)}W"
    if seconds % (60 * 60 * 24) == 0:
        return f"{seconds // (60 * 60 * 24)}D"
    if seconds % 60 == 0:
        return f"{seconds // 60}"
    return f"{seconds}S"


def _in_seconds(timeframe: str | None = None) -> int:
    """
    Convert the timeframe to seconds

    :param timeframe: The timeframe to convert. If not provided or an empty string,
        uses the current chart timeframe.
    :return: The timeframe in seconds
    :raises InvalidTimeframeError: If the timeframe is invalid
    :raises TickTimeframeError: If the timeframe is a tick timeframe
    """
    if not timeframe:
        timeframe: str = _current_period()
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
        raise _TickTimeframeError(timeframe=timeframe)


def in_seconds(timeframe: str | None = None) -> PyneInt:
    """
    Convert timeframe to seconds

    :param timeframe: The timeframe string, if None the current timeframe is used
    :return: The timeframe in seconds
    """
    # The core keeps the native int form (:func:`_in_seconds`)
    return pine_int(_in_seconds(timeframe))


@module_property
def isdaily() -> bool:
    """
    Check if the current timeframe is daily

    :return: True if the current timeframe is daily
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == 'D'


@module_property
def isdwm() -> bool:
    """
    Check if the current timeframe is intraday, daily, weekly or monthly

    :return: True if the current timeframe is intraday, daily, weekly or monthly
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == 'D' or modifier == 'W' or modifier == 'M'


@module_property
def isintraday() -> bool:
    """
    Check if the current timeframe is intraday

    :return: True if the current timeframe is intraday
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == '' or modifier == 'S' or modifier == 'T'


@module_property
def isminutes() -> bool:
    """
    Check if the current timeframe is minutes

    :return: True if the current timeframe is minutes
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == ''


@module_property
def ismonthly() -> bool:
    """
    Check if the current timeframe is monthly

    :return: True if the current timeframe is monthly
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == 'M'


@module_property
def isseconds() -> bool:
    """
    Check if the current timeframe is seconds

    :return: True if the current timeframe is seconds
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == 'S'


@module_property
def isticks() -> bool:
    """
    Check if the current timeframe is ticks

    :return: True if the current timeframe is ticks
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == 'T'


@module_property
def isweekly() -> bool:
    """
    Check if the current timeframe is weekly

    :return: True if the current timeframe is weekly
    """
    modifier, _ = _process_tf(_current_period())
    return modifier == 'W'


# noinspection PyProtectedMember
@module_property
def main_period() -> str:
    """
    Get the main period

    :return: The main period
    """
    # Inside a request.security child ``_script`` is None, but the main period
    # must still report the chart TF (propagated via ``_main_timeframe``), not the
    # context's own period, so ``timeframe.change(timeframe.main_period)`` rolls
    # over on chart bars rather than on every intrabar.
    if lib._main_timeframe is not None:
        return lib._main_timeframe
    return _current_period()


@module_property
def multiplier() -> PyneInt:
    """
    Get the current timeframe multiplier

    :return: The current timeframe multiplier
    """
    _, _multiplier = _process_tf(_current_period())
    return pine_int(_multiplier)


@module_property
def period() -> str:
    """
    Get the current period

    :return: The current period
    """
    return _current_period()
