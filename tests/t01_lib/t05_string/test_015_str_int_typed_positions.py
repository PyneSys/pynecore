"""
``str.substring`` and ``str.replace`` consume int-TYPED positions.

Pine's ``int`` is a static type only: ``(R + z) / 8`` is int-typed and carries
the value 1.75. TradingView truncates it where the position is used, while
PyneCore handed it to a slice and to ``range()`` bare and raised.

MEASURED on TradingView (FX:EURUSD@60, ``d = (R + z) / 8`` = 1.75,
``R = input.int(14)``):

| expression                           | TradingView | PyneCore before             |
|--------------------------------------|-------------|-----------------------------|
| `str.substring("abcdef", d, d + 2)`  | `bc`        | TypeError: slice indices    |
| `str.replace("aXbXcX", "X", "-", d)` | `aXb-cX`    | TypeError: 'float' object   |
"""
from pynecore.lib import string


def __test_substring_truncates_positions__():
    """A fractional begin/end position slices like its truncated integer"""
    assert string.substring("abcdef", 1.75, 3.75) == "bc"
    assert string.substring("abcdef", 1.75, 3.75) == string.substring("abcdef", 1, 3)
    # The int-typed division that reaches this in Pine code
    assert string.substring("abcdef", 14 / 8, 14 / 8 + 2) == "bc"
    # Without an end position the tail is taken from the truncated begin
    assert string.substring("abcdef", 2.75) == string.substring("abcdef", 2) == "cdef"


def __test_substring_empty_range_uses_truncated_positions__():
    """The empty-slice shortcut compares the same integers the slice would"""
    # 1.75 and 1.9 are different floats but the same position
    assert string.substring("abcdef", 1.75, 1.9) == ""
    assert string.substring("abcdef", 0.0, 0.5) == ""


def __test_replace_truncates_the_occurrence__():
    """The nth occurrence is counted with the truncated occurrence number"""
    assert string.replace("aXbXcX", "X", "-", 1.75) == "aXb-cX"
    assert string.replace("aXbXcX", "X", "-", 1.75) == string.replace("aXbXcX", "X", "-", 1)
    # An empty target is an insertion point, clamped to the end of the source
    assert string.replace("abc", "", "-", 2.75) == string.replace("abc", "", "-", 2) == "ab-c"
    assert string.replace("abc", "", "-", 4.75) == "abc-"
