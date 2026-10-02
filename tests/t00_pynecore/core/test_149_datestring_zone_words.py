"""
Zone handling of ``core.datetime.parse_datestring`` (``timestamp(dateString)``).

Verified live on TradingView, NASDAQ:AAPL daily (exchange timezone
America/New_York), so a result in the exchange timezone and one in UTC differ:

- A date string without a zone is UTC, never the exchange timezone:
  ``"01 Jan 2022 00:00"``, ``"2022-01-01"``, ``"2022"``, ``"Jan 2022"`` all give
  1640995200000.
- Words in front of the date are skipped: ``"UTC 01 Jan 2022 00:00"``, ``"EST ..."``,
  ``"America/New_York ..."``, ``"Sat, ..."`` and ``"hello world ..."`` all give
  1640995200000, ``"UTC 01 Jan 2022 00:00 +0300"`` gives 1640984400000. A glued
  word (``"UTC01 Jan 2022 00:00"``) and an ISO date/time behind a word
  (``"UTC 2022-01-01T05:00"``) are rejected.
- A trailing RFC 2822 zone name or ``Z`` is honoured in any letter case
  (``"EST"`` / ``"est"`` -> 1641013200000, ``"PDT"`` -> 1641020400000); ``"CET"``,
  ``"A"`` and an IANA name are rejected.
- A date spelled with a month name and no time ignores its zone
  (``"01 Jan 2022 PST"``, ``"01 Jan 2022 GMT+3"`` -> 1640995200000) and rejects a
  bare offset (``"01 Jan 2022 +0300"``); a numeric date keeps it
  (``"2022-01-01 EST"`` -> 1641013200000).

The wild corpus strategy "[MT] Strategy Backtest Template" failed its run on
``timestamp('UTC 01 Jan 2022 00:00')`` with "Invalid date format".
"""
import pytest

from pynecore import lib
from pynecore.core.datetime import parse_datestring

MIDNIGHT_UTC = 1640995200000


@pytest.fixture(autouse=True)
def _new_york_exchange():
    saved = lib.syminfo.timezone
    lib.syminfo.timezone = "America/New_York"
    yield
    lib.syminfo.timezone = saved


def _millis(datestring: str) -> int:
    return round(parse_datestring(datestring).timestamp() * 1000)


def __test_no_zone_is_utc_not_the_exchange_zone__():
    """A zone-less date string is UTC whatever the exchange timezone is"""
    for datestring in ("01 Jan 2022 00:00", "2022-01-01", "2022", "Jan 2022",
                       "Jan 01 2022", "2022-01-01 00:00", "2022-01-01T00:00"):
        assert _millis(datestring) == MIDNIGHT_UTC, datestring
    assert _millis("01-02-2022") == 1641081600000


def __test_leading_words_are_skipped__():
    """Words before the date carry no zone"""
    for datestring in ("UTC 01 Jan 2022 00:00", "GMT 01 Jan 2022 00:00", "EST 01 Jan 2022 00:00",
                       "America/New_York 01 Jan 2022 00:00", "Sat, 01 Jan 2022 00:00",
                       "hello world 01 Jan 2022 00:00", "UTC Jan 01 2022", "UTC Jan 2022",
                       "UTC 01 Jan 2022", "UTC 2022-01-01"):
        assert _millis(datestring) == MIDNIGHT_UTC, datestring
    assert _millis("UTC 01-02-2022") == 1641081600000
    assert _millis("UTC 01 Jan 2022 00:00 +0300") == 1640984400000
    assert _millis("Sat, 01 Jan 2022 00:00 EST") == 1641013200000


def __test_trailing_zone_names__():
    """RFC 2822 zone names and Z after the date are their offsets"""
    expected = {
        "UT": MIDNIGHT_UTC, "UTC": MIDNIGHT_UTC, "utc": MIDNIGHT_UTC, "GMT": MIDNIGHT_UTC,
        "Z": MIDNIGHT_UTC, "EST": 1641013200000, "est": 1641013200000, "EDT": 1641009600000,
        "CST": 1641016800000, "CDT": 1641013200000, "MST": 1641020400000,
        "MDT": 1641016800000, "PST": 1641024000000, "PDT": 1641020400000,
    }
    for name, millis in expected.items():
        assert _millis(f"01 Jan 2022 00:00 {name}") == millis, name
    assert _millis("01 Jan 2022 00:00:00 EST") == 1641013200000
    assert _millis("2022-01-01 00:00 EST") == 1641013200000


def __test_half_hour_offset_is_kept__():
    """A trailing offset keeps its minutes"""
    assert _millis("01 Jan 2022 00:00 UTC+0530") == 1640975400000
    assert _millis("01 Jan 2022 00:00 GMT+2") == 1640988000000


def __test_month_name_date_without_time_ignores_its_zone__():
    """Only a numeric date keeps a zone when no time is given"""
    for datestring in ("01 Jan 2022 EST", "01 Jan 2022 PST", "01 Jan 2022 GMT+3",
                       "Jan 01 2022 EST", "01 Jan 2022 UTC"):
        assert _millis(datestring) == MIDNIGHT_UTC, datestring
    assert _millis("2022-01-01 EST") == 1641013200000
    assert _millis("01-02-2022 EST") == 1641099600000


def __test_rejected_forms__():
    """Forms TradingView refuses to compile must raise here too"""
    for datestring in ("UTC01 Jan 2022 00:00", "UTC 2022-01-01T05:00", "01 Jan 2022 00:00 CET",
                       "01 Jan 2022 00:00 A", "01 Jan 2022 00:00 America/New_York",
                       "01 Jan 2022 +0300"):
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_datestring(datestring)
