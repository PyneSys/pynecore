"""
Colon-less timezone offsets in ``core.datetime.parse_datestring``.

TradingView's ``timestamp(dateString)`` takes the offset of an ISO date/time as
four digits with or without a colon. Verified live on TradingView
(CAPITALCOM:EURUSD, 30m): ``timestamp("2026-08-01T11:00:00-0500")`` and
``"2026-08-01T11:00:00-05:00"`` both give 1785600000000, as do the second-less
``"2026-08-01T11:00-0500"`` and the space-separated ``"2026-08-01 11:00:00-0500"``;
``"2026-08-01T11:00:00+0530"`` gives 1785562200000 and a fraction survives
(``"2026-08-01T11:00:00.250-0500"`` -> 1785600000250). An hour-only offset
(``"-05"``, ``"-5"``), ``"Z"`` and an offset behind a space are rejected at
compile time with "timestamp(s): unrecognized datetime format". A corpus
indicator's ``input.time(timestamp("2026-08-01T11:00:00-0500"))`` previously
failed the run with "Invalid date format".
"""
import pytest

from pynecore.core.datetime import parse_datestring


def _millis(datestring: str) -> int:
    return round(parse_datestring(datestring).timestamp() * 1000)


def __test_compact_offset_matches_the_colon_form__():
    """An offset without a colon is the same offset"""
    for datestring in ("2026-08-01T11:00:00-0500", "2026-08-01T11:00:00-05:00",
                       "2026-08-01T11:00-0500", "2026-08-01 11:00:00-0500"):
        assert _millis(datestring) == 1785600000000, datestring


def __test_compact_offset_with_minutes__():
    """The minute part of a compact offset is honoured"""
    assert _millis("2026-08-01T11:00:00+0530") == 1785562200000


def __test_compact_offset_after_a_fraction__():
    """A fractional second keeps its place before a compact offset"""
    assert _millis("2026-08-01T11:00:00.250-0500") == 1785600000250


def __test_malformed_offsets_are_rejected__():
    """Forms TradingView refuses to compile must raise here too"""
    for datestring in ("2026-08-01T11:00:00-05", "2026-08-01T11:00:00-5",
                       "2026-08-01T11:00:00Z", "2026-08-01T11:00:00 -0500"):
        with pytest.raises(ValueError, match="Invalid date format"):
            parse_datestring(datestring)
