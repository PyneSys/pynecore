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
    """ str.format_time(na) returns na (Pine na-propagation), not a string """
    from pynecore.lib import string as _string
    from pynecore.types.na import NA
    assert isinstance(_string.format_time(NA(int), "yyyy-MM-dd HH:mm"), NA)
    assert isinstance(_string.format_time(None), NA)
