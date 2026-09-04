from typing import Any

from ..types.datetime import DayOfWeek
from ..types.na import na_int
from ..types.pine_types import PyneInt
from ..core.module_property import module_function_property


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
# Module function
#

# ``pynecore.lib`` imports this module, so the bind cannot move to the top of the
# file -- it would import a half-built package. Binding it on first call keeps the
# ordering as it is and drops the import machinery from every later bar.
_lib: Any = None


# noinspection PyShadowingNames,PyProtectedMember
@module_function_property
def dayofweek(time: int | float | None = None, timezone: str | None = None) -> PyneInt:
    """
    Day of the week

    :param time: The time to get the day of the week from, if None the current time is used
    :param timezone: The timezone of the time, if not specified the exchange timezone is used
    :return: The day of the week, 1 is Sunday, 2 is Monday, ..., 7 is Saturday
    """
    global _lib
    if (lib := _lib) is None:
        from .. import lib
        _lib = lib
    dt = lib._get_dt(time, timezone)
    if dt is None:
        return na_int
    res = dt.weekday() + 2
    if res == 8:
        res = 1
    return float(res)
