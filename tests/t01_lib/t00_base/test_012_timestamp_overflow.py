"""
@pyne
"""
from pynecore.lib import timestamp


def __test_timestamp_hour_overflow__():
    """ timestamp() rolls an out-of-range hour over (Pine semantics): hour 26
    on day 11 equals 02:00 on day 12 """
    assert timestamp(2026, 6, 11, 26, 0) == timestamp(2026, 6, 12, 2, 0)


def __test_timestamp_minute_second_overflow__():
    """ minute/second overflow carries into the higher units """
    assert timestamp(2026, 6, 11, 10, 75) == timestamp(2026, 6, 11, 11, 15)
    assert timestamp(2026, 6, 11, 10, 0, 90) == timestamp(2026, 6, 11, 10, 1, 30)


def __test_timestamp_day_overflow__():
    """ a day past the month end rolls into the next month """
    assert timestamp(2026, 1, 32, 0, 0) == timestamp(2026, 2, 1, 0, 0)


def __test_timestamp_month_overflow__():
    """ month 13 rolls into the next January """
    assert timestamp(2026, 13, 1, 0, 0) == timestamp(2027, 1, 1, 0, 0)


def __test_timestamp_in_range_unchanged__():
    """ in-range components are unaffected by the overflow normalization """
    assert timestamp("UTC", 2025, 1, 1, 1, 23, 45) == timestamp("UTC", 2025, 1, 1, 1, 23, 45)
    # UTC midnight 2025-01-01 in epoch ms
    assert timestamp("UTC", 2025, 1, 1, 0, 0, 0) == 1735689600000
