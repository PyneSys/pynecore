"""
Colon-separated date/time strings in ``core.datetime.parse_datestring``.

TradingView's ``timestamp(dateString)`` accepts a colon between the date and the
time on top of the documented "T" and space separators, and the hour needs no
zero padding. Verified live on TradingView (BINANCE:BTCUSDT, 30m):
``timestamp("2021-01-13:05:00")``, ``"2021-01-13:5:00"``,
``"2021-01-13:05:00:00"`` and ``"2021-01-13T05:00"`` all resolve to
2021-01-13 05:00 UTC, and ``"2021-01-13:05:00+02:00"`` honours the offset
(03:00 UTC). A missing separator (``"2021-01-1305:00"``), a letter one
(``"2021-01-13x05:00"``) and a minute-less time (``"2021-01-13:05"``) are all
rejected at compile time with "timestamp(s): unrecognized datetime format", so
they must raise here too. A corpus strategy's
``input(type=input.time, defval=timestamp("2021-01-13:05:00"))`` previously
failed the run with "Invalid date format".
"""
from datetime import datetime

import pytest

from pynecore.core.datetime import parse_datestring


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None)


def __test_colon_separator_matches_t_and_space__():
    """A colon between date and time is the same separator as "T" and " " """
    expected = datetime(2021, 1, 13, 5, 0)
    for datestring in ("2021-01-13:05:00", "2021-01-13T05:00", "2021-01-13 05:00"):
        assert _naive(parse_datestring(datestring)) == expected, datestring


def __test_colon_separator_unpadded_hour__():
    """The hour may be a single digit"""
    assert _naive(parse_datestring("2021-01-13:5:00")) == datetime(2021, 1, 13, 5, 0)


def __test_colon_separator_with_seconds__():
    """Seconds are optional after the colon separator"""
    assert _naive(parse_datestring("2021-01-13:05:00:00")) == datetime(2021, 1, 13, 5, 0)
    assert _naive(parse_datestring("2021-01-13:05:00:30")) == datetime(2021, 1, 13, 5, 0, 30)


def __test_colon_separator_with_offset__():
    """An explicit offset applies to the colon form as well"""
    dt = parse_datestring("2021-01-13:05:00+02:00")
    assert dt.utcoffset().total_seconds() == 7200
    assert int(dt.timestamp()) == 1610506800


def __test_malformed_separators_are_rejected__():
    """Forms TradingView refuses to compile must raise here too"""
    for datestring in ("2021-01-1305:00", "2021-01-13x05:00", "2021-01-13:05"):
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_datestring(datestring)
