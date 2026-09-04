"""
``array.new_int`` STORES an int-typed value, it does not truncate it.

The consumer-slot law has two halves: an integer-consuming slot truncates, a
storage slot preserves. TradingView keeps the fractional value an int-typed
expression carries all the way through an ``array<int>``, so the inverse of the
truncation tests belongs here.

MEASURED on TradingView (FX:EURUSD@60, ``d = (R + z) / 8`` = 1.75):
``array.new_int(4, d)`` then ``array.get(id, 0)`` is ``1.75``. PyneCore raised
``AssertionError: Initial value must be int!`` instead.

The SIZE argument is the other half: it is consumed as a count, so it truncates.
"""
from pynecore.lib import array


def __test_new_int_keeps_the_fraction__():
    """The stored value survives unchanged, exactly as on TradingView"""
    assert array.get(array.new_int(4, 7 / 4), 0) == 1.75
    assert array.get(array.new_int(4, 1.75), 3) == 1.75


def __test_new_int_truncates_the_size__():
    """The size is a count, so it is consumed as an integer"""
    assert array.size(array.new_int(7 / 4)) == array.size(array.new_int(1)) == 1
    assert array.size(array.new_int(9 / 2)) == 4
