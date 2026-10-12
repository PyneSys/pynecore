"""
@pyne
"""
from pynecore.lib import script, log, bar_index, string, timestamp


@script.indicator(title="String Format Time", shorttitle="strfmt_time")
def main():
    if bar_index == 0:
        # format_time() tests
        log.info("{0}", string.format_time(timestamp("2025-01-01 01:23:45-05:00"), "yyyy-MM-dd HH:mm",
                                           "UTC-4"))  # 2025-01-01 02:23
        log.info("{0}", string.format_time(timestamp("2025-01-01 01:23:45-05:00"), "HH:mm:ss", "UTC"))  # 06:23:45
        log.info("{0}", string.format_time(timestamp("2025-01-01 01:23:45-05:00"), "yyyy/MM/dd", "UTC+1"))  # 2025/01/01
        log.info("{0}", string.format_time(timestamp("2025-01-01 01:23:45-05:00"), "dd-MM-yyyy HH:mm",
                                           "UTC-8"))  # 31-12-2024 22:23
        # Test without specifying format
        log.info("{0}", string.format_time(timestamp("2025-01-01 01:23:45-05:00")))  # Default format


# noinspection PyShadowingNames
def __test_str_format_time__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ string.format_time() """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter, syminfo_override={
        'timezone': 'UTC-5',
    }).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)


def __test_str_format_time_na__():
    """ str.format_time(na) formats the epoch (MEASURED on TradingView, v5 and v6: a na
    time gives ``1970-01-01 00:00:00.000`` in UTC and ``1969-12-31T19:00:00-0500`` with the
    default format in America/New_York, and the result is not na) """
    from pynecore.lib import string as _string
    from pynecore.types.na import NA
    assert _string.format_time(NA(int), "yyyy-MM-dd HH:mm:ss.SSS", "UTC") \
        == "1970-01-01 00:00:00.000"
    assert _string.format_time(None, timezone="America/New_York") == "1969-12-31T19:00:00-0500"


def __test_str_format_time_pine_argument_names__():
    """ str.format_time names its arguments ``format`` and ``timezone``, the way Pine does,
    so a compiled named argument binds instead of raising TypeError """
    from pynecore.lib import string as _string
    assert _string.format_time(time=1640995200000, format="yyyy-MM-dd",
                               timezone="UTC") == "2022-01-01"
    assert _string.format_time(1640995200000, format="HH:mm", timezone="UTC+2") == "02:00"


def __test_str_format_time_pattern_letters__():
    """ str.format_time is Java's SimpleDateFormat in the US locale (MEASURED on TradingView,
    2025-03-07 14:05:09.042 UTC) """
    import calendar
    from pynecore.lib import string as _string
    t = calendar.timegm((2025, 3, 7, 14, 5, 9)) * 1000 + 42
    assert _string.format_time(t, "SSS|S|SSSS", "UTC") == "042|42|0042"
    assert _string.format_time(t, "yyyy 'at' HH 'o''clock' ''", "UTC") == "2025 at 14 o'clock '"
    assert _string.format_time(t, "E|EEE|EEEE|MMM|MMMM|M|d|y|yy|yyy", "UTC") \
        == "Fri|Fri|Friday|Mar|March|3|7|2025|25|2025"
    assert _string.format_time(t, "a|hh|h|k|K|D|z|zzzz|Z|X", "America/New_York") \
        == "AM|09|9|9|9|66|EST|Eastern Standard Time|-0500|-05"
    assert _string.format_time(t, "w|u|F|G|%|W|L|LLL|XXX|Y", "UTC") == "10|5|1|AD|%|2|3|Mar|Z|2025"


def __test_str_format_time_unquoted_letter_fails__():
    """ An unquoted letter that is no pattern letter is a run-time error (MEASURED:
    ``yyyy-MM-ddTHH:mm:ssZ`` fails with "Invalid `format` value") """
    import pytest
    from pynecore.lib import string as _string
    with pytest.raises(ValueError):
        _string.format_time(0, "yyyy-MM-ddTHH:mm:ssZ", "UTC")


def __test_str_format_time_zone_names__():
    """ Zone names are Java's, short and long, in winter and in summer (MEASURED on
    TradingView, 2025-01-15 and 2025-07-15 12:00 UTC) """
    import calendar
    from pynecore.lib import string as _string
    winter = calendar.timegm((2025, 1, 15, 12, 0, 0)) * 1000
    summer = calendar.timegm((2025, 7, 15, 12, 0, 0)) * 1000
    expected = {
        'America/New_York': ('EST|Eastern Standard Time', 'EDT|Eastern Daylight Time', '-0500|-0500|-05:00'),
        'America/Chicago': ('CST|Central Standard Time', 'CDT|Central Daylight Time', '-0600|-0600|-06:00'),
        'America/Phoenix': ('MST|Mountain Standard Time', 'MST|Mountain Standard Time', '-0700|-0700|-07:00'),
        'Europe/London': ('GMT|Greenwich Mean Time', 'BST|British Summer Time', '+0000|Z|Z'),
        'Europe/Berlin': ('CET|Central European Standard Time', 'CEST|Central European Summer Time',
                          '+0100|+0100|+01:00'),
        'Asia/Shanghai': ('CST|China Standard Time', 'CST|China Standard Time', '+0800|+0800|+08:00'),
        'Asia/Singapore': ('SGT|Singapore Standard Time', 'SGT|Singapore Standard Time', '+0800|+0800|+08:00'),
        'Asia/Kolkata': ('IST|India Standard Time', 'IST|India Standard Time', '+0530|+0530|+05:30'),
        'Australia/Sydney': ('AEDT|Australian Eastern Daylight Time', 'AEST|Australian Eastern Standard Time',
                             '+1100|+1100|+11:00'),
        'UTC': ('GMT|Greenwich Mean Time', 'GMT|Greenwich Mean Time', '+0000|Z|Z'),
        'Etc/UTC': ('UTC|Coordinated Universal Time', 'UTC|Coordinated Universal Time', '+0000|Z|Z'),
        'UTC+2': ('GMT+02:00|GMT+02:00', 'GMT+02:00|GMT+02:00', '+0200|+0200|+02:00'),
        'GMT-4': ('GMT-04:00|GMT-04:00', 'GMT-04:00|GMT-04:00', '-0400|-0400|-04:00'),
    }
    for tz, (in_winter, in_summer, offsets) in expected.items():
        assert _string.format_time(winter, "z|zzzz", tz) == in_winter, tz
        assert _string.format_time(summer, "z|zzzz", tz) == in_summer, tz
        assert _string.format_time(winter, "Z|XX|XXX", tz) == offsets, tz


def __test_str_format_time_minute_offset__():
    """ A UTC/GMT offset with minutes keeps its minutes """
    from pynecore.lib import string as _string
    assert _string.format_time(0, "HH:mm z", "GMT+0530") == "05:30 GMT+05:30"
    assert _string.format_time(0, "yyyy-MM-dd HH:mm", "UTC-5:30") == "1969-12-31 18:30"
