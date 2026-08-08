"""
@pyne

Direct unit tests for the binary search index laws, measured on TradingView.
"""
from pynecore.lib import array


def main():
    """ Dummy main to keep this a valid Pyne script """


# Measured on TradingView (CAPITALCOM:EURUSD 60) with the array below, which
# carries duplicates so leftmost and rightmost separate:
#   val:        5    10   15   20   25   30   40   45
#   search:    -1     0   -1    2   -1    4    5   -1
#   leftmost:   0     0    0    1    3    4    5    5
#   rightmost:  0     0    1    3    4    4    5    6
_ARRAY = [10, 20, 20, 20, 30, 40]
_MEASURED = {
    5: (-1, 0, 0),
    10: (0, 0, 0),
    15: (-1, 0, 1),
    20: (2, 1, 3),
    25: (-1, 3, 4),
    30: (4, 4, 4),
    40: (5, 5, 5),
    45: (-1, 5, 6),
}


def __test_binary_search_matches_tradingview__():
    """ Every measured probe value lands on TradingView's index """
    for val, (expected, _, _) in _MEASURED.items():
        assert array.binary_search(_ARRAY, val) == expected, val


def __test_binary_search_leftmost_matches_tradingview__():
    """ A hit answers the first duplicate, a miss the last smaller element """
    for val, (_, expected, _) in _MEASURED.items():
        assert array.binary_search_leftmost(_ARRAY, val) == expected, val


def __test_binary_search_rightmost_matches_tradingview__():
    """ A hit answers the last duplicate, a miss the first greater element """
    for val, (_, _, expected) in _MEASURED.items():
        assert array.binary_search_rightmost(_ARRAY, val) == expected, val


def __test_leftmost_clamps_below_the_array__():
    """
    A value under every element yields 0, never -1.

    The unclamped -1 reached ``array.get()``, whose Python indexing turned it
    into the LAST element -- silently answering the far end of the array. The
    TradingView RelativeValue library depends on this bound: it guards with
    ``size - 1 >= index`` and reads ``data.get(index)`` straight after.
    """
    assert array.binary_search_leftmost([10, 20, 30], 5) == 0
    assert array.binary_search_leftmost([10], 5) == 0


def __test_rightmost_may_point_past_the_end__():
    """ A value above every element yields the size, which the caller must guard """
    assert array.binary_search_rightmost([10, 20, 30], 45) == 3
    assert array.binary_search_rightmost([10], 45) == 1
