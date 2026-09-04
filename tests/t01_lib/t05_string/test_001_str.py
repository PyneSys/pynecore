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
    """ str.contains(na, ...) answers the bool na (na with the three-state bool, false
    without it) and never iterates the na argument forever; a real source still
    searches normally """
    from pynecore.lib import string as _string
    from pynecore.types.na import NA, na_bool, set_bool_na
    assert _string.contains(NA(str), "Closes") is False
    assert _string.contains("Market Closes", NA(str)) is False
    set_bool_na(True)
    try:
        assert _string.contains(NA(str), "Closes") is na_bool
        assert _string.contains("Market Closes", NA(str)) is na_bool
    finally:
        set_bool_na(False)
    assert _string.contains("Market Closes", "Closes") is True
    assert _string.contains("Market Open", "Closes") is False


def __test_str_replace_occurrence__():
    """ str.replace picks the nth occurrence with an overlapping scan, keeps the source
    when there is no nth one, and treats an empty target as an insertion point """
    from pynecore.lib import string as _string
    # The parameter is named the way Pine names it, so a compiled named argument binds
    assert _string.replace("aXbXcXd", "X", "-", occurrence=1) == "aXb-cXd"
    # Nth occurrence, 0-based (measured on TradingView)
    assert _string.replace("aXbXcXd", "X", "-") == "a-bXcXd"
    assert _string.replace("aXbXcXd", "X", "-", 0) == "a-bXcXd"
    assert _string.replace("aXbXcXd", "X", "-", 2) == "aXbXc-d"
    # There is no fourth X, so the source is returned unchanged
    assert _string.replace("aXbXcXd", "X", "-", 3) == "aXbXcXd"
    assert _string.replace("abc", "X", "-", 0) == "abc"
    # Overlapping occurrences: "aa" starts at index 0, 1 and 2 of "aaaa"
    assert _string.replace("aaaa", "aa", "-", 0) == "-aa"
    assert _string.replace("aaaa", "aa", "-", 1) == "a-a"
    assert _string.replace("aaaa", "aa", "-", 2) == "aa-"
    assert _string.replace("aaaa", "aa", "-", 3) == "aaaa"
    # An empty target inserts at the nth position, clamped to the end
    assert _string.replace("abc", "", "-", 0) == "-abc"
    assert _string.replace("abc", "", "-", 2) == "ab-c"
    assert _string.replace("abc", "", "-", 3) == "abc-"
    assert _string.replace("abc", "", "-", 4) == "abc-"
    assert _string.replace("", "", "-", 1) == "-"


def __test_str_split_empty_separator__():
    """ str.split with an empty separator splits into characters, while an empty source
    still yields one empty piece """
    from pynecore.lib import string as _string
    # Measured on TradingView
    assert _string.split("abc", "") == ["a", "b", "c"]
    assert _string.split("héló", "") == ["h", "é", "l", "ó"]
    # An empty source is one empty piece, not an empty array — same as any separator
    assert _string.split("", "") == [""]
    assert _string.split("", ",") == [""]
    # A non-empty separator keeps its existing behaviour, which already matches
    assert _string.split("a,b,,c", ",") == ["a", "b", "", "c"]
    assert _string.split(",a,", ",") == ["", "a", ""]


# noinspection PyShadowingBuiltins
def __test_str_substring_argument_is_named_str__():
    """ str.contains/endswith/startswith/pos name their second argument ``str``, the way
    Pine does, so a compiled named argument binds instead of raising TypeError """
    from pynecore.lib import string as _string
    assert _string.contains(source="Hello World!", str="World") is True
    assert _string.endswith(source="Hello World!", str="!") is True
    assert _string.startswith(source="Hello World!", str="Hello") is True
    assert _string.pos(source="Hello World!", str="World") == 6
