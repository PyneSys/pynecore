"""
@pyne
"""
from pynecore.lib import script, log, bar_index, string, timestamp


@script.indicator(title="String", shorttitle="string")
def main():
    if bar_index == 0:
        # contains()
        log.info("{0}", string.contains("Hello World!", "!"))  # true
        log.info("{0}", string.contains("Hello World!", "?"))  # false
        # endswith()
        log.info("{0}", string.endswith("Hello World!", "!"))  # true
        log.info("{0}", string.endswith("Hello World!", "?"))  # false
        # format_time()
        log.info("{0}", string.format_time(timestamp("2025-01-01 01:23:45-05:00"),
                                           "yyyy-MM-dd HH:mm", "UTC-4"))
        # length()
        log.info("{0}", string.length("Hello World!"))  # 12
        # lower()
        log.info("{0}", string.lower("Hello World!"))  # hello world!
        # match()
        log.info("{0}", string.match("Hello World!", "[\\w]+"))  # Hello
        # pos()
        log.info("{0}", string.pos("Hello World!", "World"))  # 6
        # repeat()
        log.info("{0}", string.repeat("Hello ", 3))  # Hello Hello Hello
        # replace()
        log.info("{0}", string.replace("Hello World!", "World",
                                       "Pyne"))  # Hello Pyne!
        # replace_all()
        log.info("{0}", string.replace_all("Hello World! Hello World!",
                                           "World", "Pyne"))  # Hello Pyne! Hello Pyne!
        # split()
        log.info("{0}", string.split("Hello World!", " "))  # [Hello, World!]
        # startswith()
        log.info("{0}", string.startswith("Hello World!", "Hello"))  # true
        # substring()
        log.info("{0}", string.substring("Hello World!", 6, 11))  # World
        # tonumber()
        log.info("{0}", string.tonumber("123"))
        log.info("{0}", string.tonumber("12.3"))  # 12.3
        log.info("{0}", string.tonumber("abc"))  # NaN
        # tostring()
        log.info("{0}", string.tostring(123))  # 123
        log.info("{0}", string.tostring(12.3))  # 12.3
        log.info("{0}", string.tostring("abc"))  # abc
        log.info("{0}", string.tostring(True))  # true
        log.info("{0}", string.tostring(123, '#.00'))  # 123.00
        log.info("{0}", string.tostring(-123, '00000.00'))  # 00123.00
        # trim()
        log.info("{0}", string.trim("  Hello World!  "))  # Hello World!
        # upper()
        log.info("{0}", string.upper("Hello World!"))  # HELLO WORLD!


# noinspection PyShadowingNames
def __test_str__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Functions """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)


def __test_str_contains_na__():
    """ str.contains(na, ...) returns na (Pine na-propagation) and never
    iterates the na argument forever; a real source still searches normally """
    from pynecore.lib import string as _string
    from pynecore.types.na import NA
    assert isinstance(_string.contains(NA(str), "Closes"), NA)
    assert isinstance(_string.contains("Market Closes", NA(str)), NA)
    assert _string.contains("Market Closes", "Closes") is True
    assert _string.contains("Market Open", "Closes") is False
