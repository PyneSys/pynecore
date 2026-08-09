"""
@pyne

Negative indices address an array from its end, and only within its size.

Pine v6 grants what Python's own indexing already does: measured on TradingView
(FX:EURUSD 240, array [10, 20, 30, 40]) get(a, -1) is 40 under v6, while the very
same call halts on every bar under v4 and v5. Out of range stays an error in both
directions -- get(a, -5) and get(a, 4) halt with RE10045.

`insert` takes one position more (the array size itself, which appends), and its
bound needs its own check because Python's list.insert clamps instead of raising.
"""
import pytest

from pynecore.lib import array


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_negative_index_reads_from_the_end__():
    """ get/set/remove count a negative index from the end of the array """
    # TV probe: get([10, 20, 30, 40], -1) -> 40 (v6; the same call halts on v4/v5)
    a = [10, 20, 30, 40]
    assert array.get(a, -1) == 40
    assert array.get(a, -4) == 10

    # TV probe: set(a, -1, 99) -> the last element becomes 99
    array.set(a, -1, 99)
    assert a == [10, 20, 30, 99]

    # TV probe: remove(a, -1) -> returns 40 and leaves size 3
    b = [10, 20, 30, 40]
    assert array.remove(b, -1) == 40
    assert b == [10, 20, 30]


def __test_negative_index_beyond_the_size_raises__():
    """ Negative indexing is bound by the array size, exactly as the positive one is """
    # TV probe: get(a, -5) on a 4-element array -> RE10045, like get(a, 4)
    with pytest.raises(IndexError):
        array.get([10, 20, 30, 40], -5)
    with pytest.raises(IndexError):
        array.set([10, 20, 30, 40], -5, 1)
    with pytest.raises(IndexError):
        array.remove([10, 20, 30, 40], -5)


def __test_negative_index_on_a_slice_view__():
    """ A slice view is indexed from its own end, not the parent's """
    # TV probe: get(slice([10, 20, 30, 40], 1, 3), -1) -> 30
    a = [10, 20, 30, 40]
    view = array.slice(a, 1, 3)
    assert array.get(view, -1) == 30
    array.set(view, -1, 99)
    assert a == [10, 20, 99, 40]
    with pytest.raises(IndexError):
        array.get(view, -3)


def __test_insert_addresses_one_position_more__():
    """ insert takes the array size as an index, and a negative one prepends at -size """
    # TV probe: insert(a, 4, 77) -> 10,20,30,40,77 | insert(a, -4, 77) -> 77,10,20,30,40
    a = [10, 20, 30, 40]
    array.insert(a, 4, 77)
    assert a == [10, 20, 30, 40, 77]

    b = [10, 20, 30, 40]
    array.insert(b, -4, 77)
    assert b == [77, 10, 20, 30, 40]

    # TV probe: insert(a, -1, 77) -> 10,20,30,77,40, i.e. before the last element
    c = [10, 20, 30, 40]
    array.insert(c, -1, 77)
    assert c == [10, 20, 30, 77, 40]


def __test_insert_out_of_range_raises__():
    """ Past the array size in either direction insert is an error, not a clamp """
    # TV probe: insert(a, 5, 77) and insert(a, -5, 77) both halt with RE10045.
    # Python's list.insert would have clamped them to the end and the front.
    a = [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.insert(a, 5, 77)
    assert a == [10, 20, 30, 40]
    with pytest.raises(IndexError):
        array.insert(a, -5, 77)
    assert a == [10, 20, 30, 40]


def __test_insert_into_an_empty_array__():
    """ An empty array accepts index 0 only """
    empty: list[int] = []
    array.insert(empty, 0, 77)
    assert empty == [77]

    other: list[int] = []
    with pytest.raises(IndexError):
        array.insert(other, 1, 77)
    assert other == []
