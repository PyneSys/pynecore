"""
Partial (year-only / year-month) date strings in ``core.datetime.parse_datestring``.

TradingView's ``timestamp(dateString)`` accepts a date with the trailing
components left off and fills them in with the start of the period. Verified live
on TradingView (BINANCE:BTCUSDT, exchange timezone UTC): ``timestamp("2025")``
and ``timestamp("Jan 2025")`` / ``timestamp("January 2025")`` resolve to
2025-01-01 00:00, and ``timestamp("2025-06")`` together with its "2025-6",
"2025/06" and "2025.06" spellings resolve to 2025-06-01 00:00. A leading month
("06 2025") is rejected at compile time ("timestamp(s): unrecognized datetime
format"), so it must keep raising here. A corpus script's
``input.time(timestamp("2025"))`` previously failed with "Invalid date format".
"""
from datetime import datetime

import pytest

from pynecore.core.datetime import parse_datestring


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


def __test_year_only__():
    """"2025" is the first moment of January 2025"""
    dt = parse_datestring("2025")
    assert _naive(dt) == datetime(2025, 1, 1)
    assert dt.tzinfo is not None


def __test_year_month_across_separators__():
    """"2025-06" is June 1 with '-', '/' and '.' separators alike"""
    expected = datetime(2025, 6, 1)
    for datestring in ("2025-06", "2025/06", "2025.06"):
        dt = parse_datestring(datestring)
        assert _naive(dt) == expected, datestring
        assert dt.tzinfo is not None, datestring


def __test_single_digit_month__():
    """"2025-6" parses like its zero-padded form"""
    assert _naive(parse_datestring("2025-6")) == datetime(2025, 6, 1)


def __test_month_name_and_year__():
    """A month name with no day defaults to the 1st"""
    for datestring in ("Jan 2025", "January 2025"):
        assert _naive(parse_datestring(datestring)) == datetime(2025, 1, 1), datestring


def __test_month_first_partial_is_rejected__():
    """"06 2025" must raise -- TradingView rejects a leading month"""
    with pytest.raises(ValueError, match="Invalid date format"):
        parse_datestring("06 2025")


def __test_invalid_month_is_rejected__():
    """A month outside 1..12 is not a date"""
    with pytest.raises(ValueError):
        parse_datestring("2025-13")


def __test_existing_formats_unaffected__():
    """Full dates and the month-first numeric forms keep parsing as before"""
    assert _naive(parse_datestring("2025-06-15")) == datetime(2025, 6, 15)
    assert _naive(parse_datestring("03-04-2023")) == datetime(2023, 3, 4)
    assert _naive(parse_datestring("1 January 2018")) == datetime(2018, 1, 1)
