from datetime import datetime, timedelta

from ..types.session import Session

from ..core.module_property import module_property

from . import syminfo
from . import timeframe
from .. import lib

__all__ = [
    "regular",
    "extended",
    "isfirstbar_regular",
    "isfirstbar",
    "islastbar_regular",
    "islastbar",
    "ismarket",
    "ispremarket",
    "ispostmarket"
]

#
# Constants
#

regular = Session('regular')
extended = Session('extended')


#
# Functions
#

# noinspection PyProtectedMember
def _check_session(dt: datetime, tf_sec: int) -> bool:
    """
    Check if candle overlaps with any trading session.

    :param dt: Start datetime of the candle
    :param tf_sec: Timeframe in seconds
    :return: True if candle overlaps with any session
    """
    candle_end = dt + timedelta(seconds=tf_sec)
    candle_start = dt

    for day, ss, se in syminfo._opening_hours:
        if day != dt.weekday():
            continue

        ssdt = dt.replace(hour=ss.hour, minute=ss.minute, second=ss.second, microsecond=0)
        sedt = dt.replace(hour=se.hour, minute=se.minute, second=se.second, microsecond=0)
        if sedt < ssdt:  # Overnight session
            sedt += timedelta(days=1)

        # Check if candle overlaps with session
        if candle_end >= ssdt and candle_start < sedt:
            return True

    return False


#
# Module properties
#

# noinspection PyProtectedMember
@module_property
def isfirstbar_regular() -> bool:
    """
    Check if the current candle is the first of the trading session.
    The result is the same whether extended session information is used or not.

    :return: True if the current candle is the first of the trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    for ss in syminfo._session_starts:
        if ss.day == lib._datetime.weekday():
            ssdt = lib._datetime.replace(hour=ss.time.hour, minute=ss.time.minute, second=ss.time.second,
                                         microsecond=ss.time.microsecond)
            if lib._datetime <= ssdt < lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
# TODO: implement this when extended session will be supported
@module_property
def isfirstbar() -> bool:
    """
    Check if the current candle is the first of the trading session.
    If extended session information is used, only returns true on the first bar of the pre-market bars.
    NOTE: extended session information is not yet supported.

    :return: True if the current candle is the first of the trading session
    """
    # TODO: support pre market sessions
    tf_sec = timeframe.in_seconds(syminfo.period)
    for ss in syminfo._session_starts:
        if ss.day == lib._datetime.weekday():
            ssdt = lib._datetime.replace(hour=ss.time.hour, minute=ss.time.minute, second=ss.time.second,
                                         microsecond=ss.time.microsecond)
            if lib._datetime <= ssdt < lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
@module_property
def islastbar_regular() -> bool:
    """
    Check if the current candle is the last of the trading session.
    The result is the same whether extended session information is used or not.

    :return: True if the current candle is the last of the trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    for se in syminfo._session_ends:
        if se.day == lib._datetime.weekday():
            sedt = lib._datetime.replace(hour=se.time.hour, minute=se.time.minute, second=se.time.second,
                                         microsecond=se.time.microsecond)
            if lib._datetime < sedt < lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
@module_property
def islastbar() -> bool:
    """
    Check if the current candle is the last of the trading session.
    If extended session information is used, only returns true on the last bar of the post-market bars.
    NOTE: extended session information is not yet supported.

    :return: True if the current candle is the last of the trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    for se in syminfo._session_ends:
        if se.day == lib._datetime.weekday():
            sedt = lib._datetime.replace(hour=se.time.hour, minute=se.time.minute, second=se.time.second,
                                         microsecond=se.time.microsecond)
            if lib._datetime < sedt < lib._datetime + timedelta(seconds=tf_sec):
                return True
    return False


# noinspection PyProtectedMember
@module_property
def ismarket() -> bool:
    """
    Check if the current candle is within a trading session.

    :return:  True if the current candle is within a trading session
    """
    tf_sec = timeframe.in_seconds(syminfo.period)
    return _check_session(lib._datetime, tf_sec)


@module_property
def ispremarket() -> bool:
    """
    Check if the current candle is within the pre-market session.
    It is not yet implemented.

    :return: It is always False at the moment
    """
    # TODO: implement this
    return False


@module_property
def ispostmarket() -> bool:
    """
    Check if the current candle is within the post-market session.
    It is not yet implemented.

    :return: It is always False at the moment
    """
    # TODO: implement this
    return False
