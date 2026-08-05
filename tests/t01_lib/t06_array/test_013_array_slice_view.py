"""
@pyne

array.slice returns a live, writable view of the parent array.

The view used to be read-only, so every structural mutation of a slice --
push, insert, unshift, remove, pop, shift, clear -- died with
`'SequenceView' object has no attribute 'append'`. It also kept reporting its
full declared size after the parent had shrunk under it, which made a plain
read of a stale sibling slice raise IndexError.
"""
import pytest

from pynecore.lib import array, order
from pynecore.types.na import NA


def main():
    """ Dummy main to keep this a valid Pyne script """


def __test_push_lands_at_the_slice_end__():
    """ push/insert/unshift write into the parent inside the slice's own bounds """
    # Measured on TradingView (FX:EURUSD 240) on [10, 20, 30, 40] sliced [0, 2):
    #   push 99    -> parent 10,20,99,30,40   slice 10,20,99
    #   insert@1   -> parent 10,99,20,30,40   slice 10,99,20
    #   unshift 99 -> parent 99,10,20,30,40   slice 99,10,20
    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.push(s, 99)
    assert a == [10, 20, 99, 30, 40]
    assert list(s) == [10, 20, 99]

    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.insert(s, 1, 99)
    assert a == [10, 99, 20, 30, 40]
    assert list(s) == [10, 99, 20]

    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.unshift(s, 99)
    assert a == [99, 10, 20, 30, 40]
    assert list(s) == [99, 10, 20]

    # A slice in the middle proves the boundary is the slice's, not the parent's
    a = [10, 20, 30, 40]
    s = array.slice(a, 1, 3)
    array.push(s, 99)
    assert a == [10, 20, 30, 99, 40]
    assert list(s) == [20, 30, 99]


def __test_removal_cuts_the_element_out_of_the_parent__():
    """ remove/pop/shift/clear delete from the parent, within the slice """
    # Measured on TradingView on [10, 20, 30, 40] sliced [0, 2):
    #   remove@0 -> returns 10, parent 20,30,40
    #   pop      -> returns 20 (the SLICE's last element, not the parent's 40)
    #   shift    -> returns 10, parent 20,30,40
    #   clear    -> parent 30,40
    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    assert array.remove(s, 0) == 10
    assert a == [20, 30, 40]
    assert list(s) == [20]

    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    assert array.pop(s) == 20
    assert a == [10, 30, 40]

    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    assert array.shift(s) == 10
    assert a == [20, 30, 40]

    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.clear(s)
    assert a == [30, 40]
    assert array.size(s) == 0


def __test_the_view_is_an_index_range_not_a_content_reference__():
    """ Mutating the parent shifts what the view shows, it does not move the view """
    # Measured on TradingView: a parent push leaves the [0, 2) view at 10,20;
    # removing the parent's first element leaves that same view showing 20,30;
    # an unshift moves a [1, 3) view's content from 20,30 to 10,20
    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.push(a, 99)
    assert list(s) == [10, 20]

    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.remove(a, 0)
    assert list(s) == [20, 30]

    a = [10, 20, 30, 40]
    s = array.slice(a, 1, 3)
    array.unshift(a, 99)
    assert list(s) == [10, 20]


def __test_sibling_slices_shift_under_each_other__():
    """ Two slices of one parent are independent ranges, so one shifts the other """
    # Measured on TradingView with [0, 2) and [2, 4) over [10, 20, 30, 40]:
    #   push 99 into the front slice -> back slice shows 99,30
    #   push 99 into the back slice  -> front slice unchanged, back 30,40,99
    #   overlapping [0, 3) and [1, 4), push into the first -> second is 20,30,99
    a = [10, 20, 30, 40]
    front = array.slice(a, 0, 2)
    back = array.slice(a, 2, 4)
    array.push(front, 99)
    assert a == [10, 20, 99, 30, 40]
    assert list(back) == [99, 30]

    a = [10, 20, 30, 40]
    front = array.slice(a, 0, 2)
    back = array.slice(a, 2, 4)
    array.push(back, 99)
    assert a == [10, 20, 30, 40, 99]
    assert list(front) == [10, 20]
    assert list(back) == [30, 40, 99]

    a = [10, 20, 30, 40]
    first = array.slice(a, 0, 3)
    second = array.slice(a, 1, 4)
    array.push(first, 99)
    assert list(first) == [10, 20, 30, 99]
    assert list(second) == [20, 30, 99]


def __test_a_view_is_clipped_to_the_parents_current_length__():
    """ A view reaching past the shrunken parent reports only what exists """
    # Measured on TradingView: with [2, 4) over [10, 20, 30, 40], removing the
    # parent's first element leaves the view at size 1 (just 40), and clearing the
    # whole front slice leaves it at size 0 -- neither raises
    a = [10, 20, 30, 40]
    back = array.slice(a, 2, 4)
    array.remove(a, 0)
    assert array.size(back) == 1
    assert list(back) == [40]

    a = [10, 20, 30, 40]
    front = array.slice(a, 0, 2)
    back = array.slice(a, 2, 4)
    array.clear(front)
    assert array.size(back) == 0
    assert list(back) == []


def __test_clipping_is_not_permanent__():
    """ A clipped view comes back when the parent grows past its range again """
    # Measured on TradingView: the [2, 4) view shrunk to 1 element by a parent
    # removal is back to 2 elements after a parent push, and one emptied to 0 by a
    # clear returns as the parent regrows -- so the stored range is never rewritten
    a = [10, 20, 30, 40]
    back = array.slice(a, 2, 4)
    array.remove(a, 0)
    assert array.size(back) == 1
    array.push(a, 77)
    assert list(back) == [40, 77]

    a = [10, 20, 30, 40]
    front = array.slice(a, 0, 2)
    back = array.slice(a, 2, 4)
    array.clear(front)
    assert array.size(back) == 0
    array.push(a, 77)
    array.push(a, 88)
    assert list(back) == [77, 88]


def __test_a_slice_of_a_slice_writes_through_the_whole_chain__():
    """ Every view in the chain grows when the innermost one is pushed to """
    # Measured on TradingView: parent [0, 3) sliced again to [0, 2), push 99 ->
    # parent 10,20,99,30,40, middle 10,20,99,30 (grew too), inner 10,20,99
    a = [10, 20, 30, 40]
    mid = array.slice(a, 0, 3)
    inner = array.slice(mid, 0, 2)
    array.push(inner, 99)
    assert a == [10, 20, 99, 30, 40]
    assert list(mid) == [10, 20, 99, 30]
    assert list(inner) == [10, 20, 99]

    # A nested view must print its own elements, not the view object it sits on
    # (this is what str.tostring() and every log/label formatting reaches)
    assert str(mid) == '[10, 20, 99, 30]'
    assert str(inner) == '[10, 20, 99]'


def __test_element_writes_still_reach_the_parent__():
    """ set and fill through the view keep writing single values, not lists """
    # Measured on TradingView: fill on the [0, 2) slice of [10, 20, 30, 40]
    # gives 5,5,30,40
    a = [10, 20, 30, 40]
    s = array.slice(a, 0, 2)
    array.set(s, 0, 99)
    assert a == [99, 20, 30, 40]

    a = [10, 20, 30, 40]
    array.fill(array.slice(a, 0, 2), 5)
    assert a == [5, 5, 30, 40]


def __test_whole_array_operations_stay_inside_the_slice__():
    """ sort, reverse and concat applied to a view touch only the sliced range """
    # Measured on TradingView:
    #   sort [0, 3) of 30,10,20,40           -> parent 10,20,30,40
    #   reverse [0, 3) of 10,20,30,40        -> parent 30,20,10,40
    #   sort descending [1, 3)               -> parent 10,30,20,40
    #   concat [7, 8] onto the [0, 2) slice  -> parent 10,20,7,8,30,40
    a = [30, 10, 20, 40]
    array.sort(array.slice(a, 0, 3))
    assert a == [10, 20, 30, 40]

    a = [10, 20, 30, 40]
    array.reverse(array.slice(a, 0, 3))
    assert a == [30, 20, 10, 40]

    a = [10, 20, 30, 40]
    array.sort(array.slice(a, 1, 3), order.descending)
    assert a == [10, 30, 20, 40]

    a = [10, 20, 30, 40]
    array.concat(array.slice(a, 0, 2), [7, 8])
    assert a == [10, 20, 7, 8, 30, 40]


def __test_invalid_bounds_are_rejected__():
    """ Out-of-range or reversed indices are an error, not a clamped slice """
    # Measured on TradingView (FX:EURUSD 240) on [10, 20, 30, 40]:
    #   slice(a, 0, 10) / slice(a, -1, 2) / slice(a, 0, -1) / slice(a, 4, 4) -> RE10045
    #   slice(a, 3, 1)                                                      -> RE10044
    a = [10, 20, 30, 40]
    for index_from, index_to in ((0, 10), (-1, 2), (0, -1), (4, 4), (3, 1)):
        with pytest.raises(ValueError):
            array.slice(a, index_from, index_to)

    # An empty array halts even for the na defaults, which resolve to (0, 0)
    with pytest.raises(ValueError):
        array.slice([], 0, 0)
    with pytest.raises(ValueError):
        array.slice([], NA(int), NA(int))

    # The legal edges stay legal: index_to may equal the size and the indices may match
    assert list(array.slice(a, 0, 4)) == [10, 20, 30, 40]
    assert list(array.slice(a, 3, 4)) == [40]
    assert list(array.slice(a, 1, 1)) == []
