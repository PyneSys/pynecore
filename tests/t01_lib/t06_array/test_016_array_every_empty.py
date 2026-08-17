"""
@pyne

array.every on an empty array is false, not vacuously true.

Measured on TradingView (BINANCE:BTCUSDT 30, `array.new_bool()`):
`array.every(id)` and `array.some(id)` both return false for an empty array, and
`array.includes(id, true)` does too. PyneCore used to delegate straight to
Python's `all()`, which is true for an empty iterable -- a corpus strategy whose
condition list stays empty while every filter is switched off then took an entry
TradingView never takes.
"""
from pynecore.lib import array


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_every_on_empty_array_is_false__():
    """ An empty array has no true element, so every() is false """
    assert array.every([]) is False


def __test_some_on_empty_array_is_false__():
    """ some() already agreed with TradingView and must stay false """
    assert array.some([]) is False


def __test_every_on_populated_arrays__():
    """ A non-empty array keeps the all-elements-true semantics """
    assert array.every([True, True]) is True
    assert array.every([True, False]) is False
    assert array.every([False, False]) is False


def __test_every_on_numeric_arrays__():
    """ Pine's truthiness of a numeric array is unchanged by the empty-array guard """
    assert array.every([1, 2]) is True
    assert array.every([1, 0]) is False
