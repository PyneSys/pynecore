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


def __test_timestamp_year_beyond_datetime_range__():
    """ Pine puts no upper bound on the year, so neither does timestamp() """
    # A month overflow carrying past year 9999 is what real scripts hit: a
    # "no end date" input pair like year 9999 / month 31 / day 12
    assert timestamp("UTC", 9999, 31, 12, 23, 59) == 253450598340000
    assert timestamp("UTC", 10001, 7, 12, 23, 59) == 253450598340000
    assert timestamp("UTC", 9999, 12, 31, 23, 59) == 253402300740000
    assert timestamp("UTC", 275760, 9, 13, 0, 0) == 8640000000000000
    assert timestamp("UTC", 1000000, 1, 1, 0, 0) == 31494784780800000


def __test_timestamp_clock_overflow_beyond_datetime_range__():
    """ an hour/minute/second rollover past the last representable day rolls over too """
    # 10000-01-01 00:00 UTC written three other ways
    assert timestamp("UTC", 9999, 12, 31, 24, 0) == 253402300800000
    assert timestamp("UTC", 9999, 12, 31, 23, 60) == 253402300800000
    assert timestamp("UTC", 9999, 12, 31, 23, 59, 60) == 253402300800000
    assert timestamp("UTC", 10000, 1, 1, 0, 0) == 253402300800000


def __test_timestamp_julian_calendar_before_cutover__():
    """ dates before 1582-10-15 follow the Julian calendar, as TradingView's does """
    assert timestamp("UTC", 1583, 1, 1, 0, 0) == -12212553600000
    assert timestamp("UTC", 1582, 10, 15, 0, 0) == -12219292800000
    assert timestamp("UTC", 1582, 1, 1, 0, 0) == -12243225600000
    assert timestamp("UTC", 1500, 1, 1, 0, 0) == -14830992000000
    assert timestamp("UTC", 1000, 1, 1, 0, 0) == -30609792000000
    assert timestamp("UTC", 1, 1, 1, 0, 0) == -62135769600000
    assert timestamp("UTC", 0, 1, 1, 0, 0) == -62167392000000
    assert timestamp("UTC", -1, 1, 1, 0, 0) == -62198928000000
    assert timestamp("UTC", -500, 1, 1, 0, 0) == -77946192000000


def __test_timestamp_skipped_cutover_dates_are_julian__():
    """ the ten days the calendar switch skipped read as Julian dates """
    assert timestamp("UTC", 1582, 10, 5, 0, 0) == timestamp("UTC", 1582, 10, 15, 0, 0)
    assert timestamp("UTC", 1582, 10, 14, 0, 0) == -12218515200000
    assert timestamp("UTC", 1582, 10, 4, 0, 0) == -12219379200000


def __test_timestamp_leap_day_rules_per_calendar__():
    """ the Gregorian century rule applies only after the cutover """
    # 1900 is not a Gregorian leap year, so Feb 29 rolls into March 1
    assert timestamp("UTC", 1900, 2, 29, 0, 0) == -2203891200000
    assert timestamp("UTC", 1900, 2, 29, 0, 0) == timestamp("UTC", 1900, 3, 1, 0, 0)
    # 1600 is a leap year in both calendars
    assert timestamp("UTC", 1600, 2, 29, 0, 0) == -11670998400000
