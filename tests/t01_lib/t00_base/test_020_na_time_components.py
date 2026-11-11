"""
@pyne

The date components of an na time are those of the Unix epoch, never na.

Measured on TradingView (CAPITALCOM:EURUSD 60, exchange zone America/New_York) with
``int n = na``: ``year/month/dayofmonth/hour/minute/second/dayofweek/weekofyear(n)``
are 1969/12/31/19/0/0/4/1, with ``"UTC"`` 1970/1/1/0/0/0/5/1, and ``hour(n,
"Asia/Kolkata")`` / ``minute(...)`` are 5/30. So ``timestamp("UTC", year(t),
month(t), dayofmonth(t), 0, 0)`` of an na ``t`` is -86400000 (GitHub issue #111).
"""
import pytest

from pynecore import lib
from pynecore.lib import dayofmonth, hour, minute, month, second, timestamp, weekofyear, year
from pynecore.lib.dayofweek import dayofweek
from pynecore.types.na import NA


@pytest.fixture(autouse=True)
def __test_helper_new_york_exchange():
    saved = lib.syminfo.timezone
    lib.syminfo.timezone = "America/New_York"
    yield
    lib.syminfo.timezone = saved


def __test_na_time_components_exchange_zone__():
    """ an na time reads as the epoch in the exchange timezone """
    n = NA(int)
    assert (year(n), month(n), dayofmonth(n), hour(n), minute(n), second(n)) == \
           (1969, 12, 31, 19, 0, 0)
    assert dayofweek(n) == 4
    assert weekofyear(n) == 1


def __test_na_time_components_explicit_zone__():
    """ an explicit timezone applies to the epoch too """
    n = NA(int)
    assert (year(n, "UTC"), month(n, "UTC"), dayofmonth(n, "UTC"), hour(n, "UTC"),
            minute(n, "UTC"), second(n, "UTC")) == (1970, 1, 1, 0, 0, 0)
    assert dayofweek(n, "UTC") == 5
    assert weekofyear(n, "UTC") == 1
    assert (hour(n, "Asia/Kolkata"), minute(n, "Asia/Kolkata"), dayofmonth(n, "Asia/Kolkata")) == \
           (5, 30, 1)


def __test_timestamp_of_na_time_components__():
    """ the issue's script: timestamp() of the components of an na time """
    t = NA(int)
    assert timestamp("UTC", year(t), month(t), dayofmonth(t), 0, 0) == -86400000


def __test_timestamp_na_components_measured__():
    """ timestamp() itself still reads an na component as 0 (measured values) """
    n = NA(int)
    assert timestamp("UTC", n, n, n, 0, 0) == -62170156800000
    assert timestamp("UTC", 2020, n, 1, 0, 0) == 1575158400000
    assert timestamp("UTC", 2020, 1, n, 0, 0) == 1577750400000
    assert timestamp("UTC", 2020, n, n, 0, 0) == 1575072000000
    assert timestamp("UTC", n, 6, n, 0, 0) == -62154345600000
    assert timestamp("UTC", n, n, 15, 0, 0) == -62168860800000
    assert timestamp("UTC", 2020, 6, 15, 5, 7, n) == 1592197620000
    assert timestamp("UTC", 2020, 6, 15, n, n, n) == 1592179200000
    assert timestamp("America/New_York", 2020, n, 1, 0, 0) == 1575176400000
