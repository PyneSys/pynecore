"""
Numeric (month-first) date strings in ``core.datetime.parse_datestring``.

TradingView's ``timestamp(dateString)`` accepts undocumented numeric dates in
MONTH-FIRST order with '-', '/' or '.' separators and an optional time part.
Verified live on TradingView: ``timestamp("03-04-2023")`` and the '/' and '.'
variants all resolve to March 4 2023 (not April 3), single-digit fields parse
("3-4-2023"), a time part may follow ("03-04-2023 10:20[:30]"), and the
day-first "13-04-2023" is rejected at compile time ("timestamp(s):
unrecognized datetime format") -- so there is intentionally no day-first
fallback to mirror. A corpus script's ``input.time(timestamp("01-01-2023"))``
previously failed with "Invalid date format".
"""
from datetime import datetime

import pytest

from pynecore.core.datetime import parse_datestring


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


def __test_month_first_across_separators__():
    """"03-04-2023" is March 4 with '-', '/' and '.' separators alike"""
    expected = datetime(2023, 3, 4)
    for datestring in ("03-04-2023", "03/04/2023", "03.04.2023"):
        dt = parse_datestring(datestring)
        assert _naive(dt) == expected, datestring
        assert dt.tzinfo is not None, datestring


def __test_single_digit_month_and_day__():
    """"3-4-2023" parses like its zero-padded form"""
    assert _naive(parse_datestring("3-4-2023")) == datetime(2023, 3, 4)


def __test_numeric_date_with_time_part__():
    """An optional HH:MM[:SS] follows the numeric date"""
    assert _naive(parse_datestring("03-04-2023 10:20")) == datetime(2023, 3, 4, 10, 20)
    assert _naive(parse_datestring("03-04-2023 10:20:30")) == datetime(2023, 3, 4, 10, 20, 30)


def __test_day_first_is_rejected__():
    """"13-04-2023" must raise like TV's compile error -- no day-first fallback"""
    with pytest.raises(ValueError, match="Invalid date format"):
        parse_datestring("13-04-2023")


def __test_existing_formats_unaffected__():
    """ISO and Pine month-name formats keep parsing as before"""
    assert _naive(parse_datestring("2020-02-20")) == datetime(2020, 2, 20)
    assert _naive(parse_datestring("Feb 01 2020 22:10:05")) == datetime(2020, 2, 1, 22, 10, 5)
    assert _naive(parse_datestring("1 January 2018")) == datetime(2018, 1, 1)
