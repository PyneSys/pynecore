from __future__ import annotations

from ..types.datetime import DayOfWeek
from ..core.module_property import module_property
from ..core.callable_module import CallableModule


#
# Module object
#

class DayOfWeekModule(CallableModule):
    #
    # Constants
    #

    sunday = DayOfWeek()
    monday = DayOfWeek()
    tuesday = DayOfWeek()
    wednesday = DayOfWeek()
    thursday = DayOfWeek()
    friday = DayOfWeek()
    saturday = DayOfWeek()


#
# Callable module function
#

# noinspection PyShadowingNames
@module_property
def dayofweek(time: int | None = None, timezone: str | None = None) -> int:
    """
    Day of the week

    :param time: The time to get the day of the week from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The day of the week, 1 is Sunday, 2 is Monday, ..., 7 is Saturday
    """
    from .. import lib
    res = lib._get_dt(time, timezone).weekday() + 2
    if res == 8:
        res = 1
    return res


#
# Module initialization
#

DayOfWeekModule(__name__)
