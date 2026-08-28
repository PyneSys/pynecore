from typing import TypeVar, Any, cast

import builtins

# Underscored: a plain module-level name here would leak into the Pine namespace
# through the module-property registry as `array.bisect_left`.
from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right

import math
import statistics

from ..utils.sequence_view import SequenceView
# Pine's absolute comparison tolerance. Only the array operations MEASURED to
# compare tolerantly use it (see core/pine_compare.py); ``binary_search``,
# ``max``/``min``, ``mode``, ``median`` and the percentile functions were
# measured bit-exact and must stay that way.
from ..core.pine_compare import EPSILON as _EPSILON, equal as _equal

from ..types.na import NA, na_float
from ..types.color import Color
from ..types.box import Box
from ..types.line import Line
from ..types.label import Label
from ..types.linefill import LineFill
from ..types.table import Table
from . import order as _order

T = TypeVar('T')
Number = TypeVar('Number', int, float)

__all__ = [
    'abs',
    'avg',
    'binary_search',
    'binary_search_leftmost',
    'binary_search_rightmost',
    'clear',
    'concat',
    'copy',
    'covariance',
    'every',
    'fill',
    'first',
    'from_items',
    'get',
    'includes',
    'indexof',
    'insert',
    'join',
    'last',
    'lastindexof',
    'max',
    'median',
    'min',
    'mode',
    'new',
    'new_bool',
    'new_box',
    'new_color',
    'new_float',
    'new_int',
    'new_label',
    'new_line',
    'new_linefill',
    'new_string',
    'new_table',
    'percentile_linear_interpolation',
    'percentile_nearest_rank',
    'percentrank',
    'pop',
    'push',
    'range',
    'remove',
    'reverse',
    'set',
    'shift',
    'size',
    'slice',
    'some',
    'sort',
    'sort_indices',
    'standardize',
    'stdev',
    'sum',
    'unshift',
    'variance',
]


def _seq_sum(values: list[float] | list[int]) -> float | int:
    """
    Add the elements front to back, one at a time.

    MEASURED (BINANCE:BTCUSDT@30, 29k bars, arrays built with ``unshift`` so index
    0 is the newest element): TradingView adds from index 0 upwards with plain
    double arithmetic. Python's ``sum()`` cannot stand in for it -- since 3.12 it
    runs Neumaier compensated summation over floats, which lands a different last
    bit on most bars and is what made ``array.avg``/``array.sum`` drift.

    :param values: Numeric elements, na already removed
    :return: The running total, an int only when every element was one
    """
    total: float | int = 0
    for v in values:
        total += v
    return total


def _moments(values: list[float]) -> tuple[float, float]:
    """
    Sum of the elements and sum of their squares, both front to back.

    :param values: Numeric elements as floats, na already removed
    :return: ``(sum, sum of squares)``
    """
    p = 0.0
    q = 0.0
    for v in values:
        p += v
    for v in values:
        q += v * v
    return p, q


def _population_stats(values: list[float]) -> tuple[float, float]:
    """
    Population mean and variance in TradingView's own arithmetic.

    :param values: Numeric elements as floats, na already removed
    :return: ``(mean, variance)``; the variance is clamped at zero
    """
    length = len(values)
    p, q = _moments(values)
    mean = p / length
    return mean, builtins.max(0.0, q / length - mean * mean)


def _numeric(values: list[Number]) -> list[float]:
    """
    The array's numeric elements as floats.

    :param values: Input array
    :return: Every element that is not na, converted to float
    """
    # non-na: neither NA nor nan equals itself
    return [float(v) for v in values if v == v]


# noinspection PyShadowingBuiltins
def _na_element(id: list[Any] | SequenceView[Any]) -> Any:
    """
    Return an na value matching the array's element type.

    An empty array has no knowable element type and yields a typeless na.

    :param id: Input array
    :return: na of the array's element type
    """
    if len(id) == 0:
        return NA(None)
    head = id[0]
    # An element that is already na carries the right type; ``type()`` of it
    # would be ``NA`` itself, and ``NA(NA)`` would be wrong.
    if not (head == head):
        return head
    return NA(builtins.type(head))


# noinspection PyShadowingBuiltins
def abs(id: list[int | float]) -> list[int | float]:
    """
    Returns an array containing the absolute value of each element in the original array.

    :param id: Input array
    :return: Array containing the absolute value of each element in the original array
    """
    return [builtins.abs(v) for v in id]


# noinspection PyShadowingBuiltins
def avg(id: list[Number]) -> float:
    """
    Returns the average value of the elements in the array.

    :param id: Input array
    :return: Average value of the elements in the array, or na if the array is empty
    """
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        return na_float
    return _seq_sum(a) / len(a)


# noinspection PyShadowingBuiltins
def binary_search(id: list[Any], val: Any) -> int:
    """
    Returns the index of the specified value in the sorted array using binary search.
    If the value is not found, returns -1.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the specified value in the sorted array, or -1 if not found
    """
    low = 0
    high = len(id) - 1
    while low <= high:
        mid = (low + high) // 2
        if id[mid] == val:
            return mid
        else:
            if val < id[mid]:
                high = mid - 1
            else:
                low = mid + 1
    return -1


# noinspection PyShadowingBuiltins
def binary_search_leftmost(id: list[Any], val: Any) -> int:
    """
    Returns the index of the first occurrence of the value in the sorted array.
    If the value is not found, returns the index of the last element smaller than the
    value, or 0 when every element is greater.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the first occurrence of the value, or of the last smaller element
    """
    # Measured on TradingView with [10, 20, 20, 20, 30, 40]:
    #   5 -> 0, 10 -> 0, 15 -> 0, 20 -> 1, 25 -> 3, 30 -> 4, 40 -> 5, 45 -> 5
    # so a hit answers the FIRST of the duplicates, a miss steps one to the left,
    # and a value below the whole array is clamped to 0 rather than returning -1.
    index = _bisect_left(id, val)
    if index < len(id) and id[index] == val:
        return index
    return index - 1 if index > 0 else 0


# noinspection PyShadowingBuiltins
def binary_search_rightmost(id: list[Any], val: Any) -> int:
    """
    Returns the index of the last occurrence of the value in the sorted array.
    If the value is not found, returns the index of the first element greater than the
    value, which is the array size when every element is smaller.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the last occurrence of the value, or of the first greater element
    """
    # Measured on TradingView with [10, 20, 20, 20, 30, 40]:
    #   5 -> 0, 10 -> 0, 15 -> 1, 20 -> 3, 25 -> 4, 30 -> 4, 40 -> 5, 45 -> 6
    # so a hit answers the LAST of the duplicates and a miss steps to the first
    # greater element -- past the end for a value above the whole array.
    index = _bisect_right(id, val)
    if index > 0 and id[index - 1] == val:
        return index - 1
    return index


# noinspection PyShadowingBuiltins
def clear(id: list[Any] | SequenceView[Any]) -> None:
    """
    Removes all elements from the array.

    :param id: Input array
    """
    id.clear()


# noinspection PyShadowingBuiltins
def concat(id1: list[T] | SequenceView[T], id2: list[T] | SequenceView[T]) \
        -> list[T] | SequenceView[T]:
    """
    Concatenates two arrays into a single array.

    :param id1: First array
    :param id2: Second array
    :return: Array containing the elements of both input arrays
    """
    id1.extend(id2)
    return id1


# noinspection PyShadowingBuiltins
def copy(id: list[T]) -> list[T]:
    """
    Returns a shallow copy of the array.

    :param id: Input array
    :return: Shallow copy of the array
    """
    return list(id)


# noinspection PyShadowingBuiltins
def covariance(id1: list[Number], id2: list[Number], biased: bool = True) -> float:
    """
    Returns the covariance between the elements in the two arrays.

    :param id1: First input array
    :param id2: Second input array
    :param biased: If True, calculates the biased covariance. If False, calculates the unbiased covariance.
    :return: Covariance between the elements in the two arrays, or na if the arrays are empty
    """
    assert len(id1) == len(id2), "Input arrays must have the same length!"
    pairs = [(float(v1), float(v2)) for v1, v2 in zip(id1, id2)
             if v1 == v1 and v2 == v2]
    if not pairs:
        return na_float
    # MEASURED (BINANCE:BTCUSDT@30, 29k bars, every bar bit-identical): unlike
    # ``variance``, the covariance is the classic TWO-PASS form -- both means
    # first, then the co-moment summed front to back -- over the divisor. A pair
    # is dropped whenever EITHER side is na, and the divisor counts the pairs
    # that survived. One surviving pair is 0.0 biased and na unbiased.
    length = len(pairs)
    if not biased and length < 2:
        return na_float
    mean1 = 0.0
    mean2 = 0.0
    for v1, v2 in pairs:
        mean1 += v1
        mean2 += v2
    mean1 /= length
    mean2 /= length
    comoment = 0.0
    for v1, v2 in pairs:
        comoment += (v1 - mean1) * (v2 - mean2)
    return comoment / (length if biased else length - 1)


# noinspection PyShadowingBuiltins
def every(id: list[Any]) -> bool:
    """
    Returns true if all elements of the id array are true, false otherwise.

    :param id: Input array
    :return: True if all elements of the id array are true, false otherwise
    """
    # Measured on TradingView: an EMPTY array yields false here, so this is not
    # Python's vacuously true all([]) -- array.some() already agrees at false.
    return len(id) > 0 and all(id)


# noinspection PyShadowingBuiltins
def fill(id: list[T] | SequenceView[T], value: T,
         index_from: int = 0, index_to: int | NA = NA(int)) -> None:
    """
    Fills the elements in the array with the specified value.

    An na ``index_from`` fills from the start of the array and an na ``index_to``
    fills to its end, instead of failing. ``index_from`` must address an existing
    element and ``index_to`` may reach one position past the last one; any other
    bound, negative ones included, is an error. Reversed bounds fill nothing. A
    slice view fills the addressed part of its parent, like every other write
    through a view.

    :param id: Input array
    :param value: Value to fill
    :param index_from: Index to start filling from
    :param index_to: Index to stop filling at
    :raises IndexError: If a bound is out of range
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100, array [10, 20, 30, 40]):
    #   fill(a, 5, na, 2) -> 5,5,30,40      fill(a, 5, 1, na) -> 10,5,5,5
    # ``index_to`` defaults to na, so the na guard also implements Pine's
    # "omitted index_to means the array size" default.
    #
    # Bounds outside the array HALT the script (RE10045), they are not clamped:
    #   fill(a, 5, 0, 99)  fill(a, 5, 0, -1)  fill(a, 5, -1, 2)  fill(a, 5, 4, 4)
    # all halt, as does fill(array.new_float(0), 5) on an empty array, whose
    # implied index_from 0 addresses nothing. So index_from addresses an element
    # while index_to may equal the size -- the range slice() takes as well. A
    # NEGATIVE bound is not counted from the end here, unlike in get/set/remove/
    # insert. Reversed bounds are a silent no-op though (fill(a, 5, 3, 1) left
    # the array untouched), where slice() halts on them.
    length = len(id)
    start = int(index_from) if index_from == index_from else 0  # is_na_arg
    stop = int(cast(int, index_to)) if index_to == index_to else length  # is_na_arg
    if not 0 <= start < length:
        raise IndexError(f"Start index {start} is out of bounds, array size is {length}")
    if not 0 <= stop <= length:
        raise IndexError(f"End index {stop} is out of bounds, array size is {length}")
    id[start:stop] = [value] * (stop - start)


# noinspection PyShadowingBuiltins
def first(id: list[T]) -> T:
    """
    Returns the first element in the array.

    :param id: Input array
    :return: First element in the array
    """
    if len(id) == 0:
        raise RuntimeError("Cannot get first element of an empty array!")
    return id[0]


# noinspection PyShadowingBuiltins
def from_items(*items: T) -> list[T]:
    """
    Returns an array containing the specified elements.
    NOTE: this is `array.from()` in Pine Script, but `from` is a reserved keyword in Python

    :param items: Elements to include in the array
    :return: Array containing the specified elements
    """
    return list(items)


# noinspection PyShadowingBuiltins
def get(id: list[T] | SequenceView[T], index: int) -> T:
    """
    Returns the element at the specified index in the array.

    An na index returns na and leaves the array untouched. A float index is
    truncated to an integer. A negative index addresses the array from its end,
    and an index outside ``-size .. size - 1`` raises.

    :param id: Input array
    :param index: Index of the element to return
    :return: Element at the specified index in the array, or na if the index is na
    :raises IndexError: If the index is out of range
    """
    # Negative indices are a Pine v6 feature and they follow Python's own rule
    # exactly, which is why nothing here handles them. Measured on TradingView
    # (FX:EURUSD 240, array [10, 20, 30, 40]): get(a, -1) is 40 while get(a, -5)
    # and get(a, 4) halt (RE10045) -- and the SAME get(a, -1) halts on every bar
    # under v4 and v5, so it is the version that grants it, not leniency here.
    # ``set``, ``remove`` and slice views (get(slice(a, 1, 3), -1) -> 30) share
    # the rule; ``insert`` extends it by one position and ``fill`` does not take
    # negative bounds at all.
    if not (index == index):  # is_na_arg
        # Measured on TradingView (FX:EURUSD 240, bar 100):
        #   get([10, 20, 30, 40], na) -> NaN, array unchanged
        #   get(array.new_int(), na)  -> NaN, size 0
        # so an empty array is tolerated too: no bounds check is reached.
        return cast(T, _na_element(id))
    # TradingView rejects a float index while compiling, so a compiled script can
    # never reach this with one. PyneCore does not always know the type, though,
    # and an integer carried as a float (from a division, from math.round) is a
    # legitimate index -- int() takes it instead of raising a TypeError that would
    # stop the script. Every index-taking function here does the same.
    return id[int(index)]


# noinspection PyShadowingBuiltins
def includes(id: list[T], value: T) -> bool:
    """
    Returns true if the array contains the specified value, false otherwise.

    The search is tolerant: a float within the float comparison tolerance of an
    element counts as present. ``binary_search`` is exact by contrast, so the two
    disagree on near-equal values.

    :param id: Input array
    :param value: Value to search for
    :return: True if the array contains the specified value, false otherwise
    """
    # Tolerance measured on TradingView (probe m548)
    for item in id:
        if _equal(item, value):
            return True
    return False


# noinspection PyShadowingBuiltins
def indexof(id: list[T], value: T) -> int:
    """
    Returns the index of the first occurrence of the specified value in the array.

    The search is tolerant, like ``includes``.

    :param id: Input array
    :param value: Value to search for
    :return: Index of the first occurrence of the specified value in the array
    """
    # Tolerance measured on TradingView (probes m548/m551)
    for i, item in enumerate(id):
        if _equal(item, value):
            return i
    return -1


# noinspection PyShadowingBuiltins
def insert(id: list[T] | SequenceView[T], index: int, value: T) -> None:
    """
    Inserts the specified value at the specified index in the array.

    An na index appends the value at the end of the array, a float index is
    truncated to an integer. A negative index counts from the end and the array
    size itself is a valid index (it appends), so ``-size .. size`` addresses a
    position and anything outside it is an error.

    :param id: Input array
    :param index: Index to insert the value at
    :param value: Value to insert
    :raises IndexError: If the index is out of range
    """
    if not (index == index):  # is_na_arg
        # Measured on TradingView (FX:EURUSD 240, bar 100):
        #   insert([10, 20, 30, 40], na, 77) -> 10,20,30,40,77
        #   insert(array.new_int(), na, 77)  -> 77
        # i.e. na resolves to the array size, it is not clamped from 0.
        id.append(value)
        return
    index = int(index)  # float-carried integer index, see get()
    # Measured on TradingView (FX:EURUSD 240, array [10, 20, 30, 40]):
    #   insert(a, 4, 77) -> 10,20,30,40,77   insert(a, -4, 77) -> 77,10,20,30,40
    #   insert(a, 5, 77) / insert(a, -5, 77) -> both halt (RE10045)
    # Python's list.insert CLAMPS an out-of-range index into the array instead of
    # raising, so this bound needs its own check -- get/set/remove inherit the
    # equivalent range from Python's own indexing and need none.
    length = len(id)
    if not -length <= index <= length:
        raise IndexError(f"Index {index} is out of bounds, array size is {length}")
    id.insert(index, value)


# noinspection PyShadowingBuiltins
def join(id: list[Any] | SequenceView[Any], separator: str) -> str:
    """
    Concatenates the elements in the array into a single string, separated by the specified separator.

    :param id: Input array
    :param separator: Separator to use
    :return: String containing the concatenated elements
    """
    sa = [str(i) for i in id]  # Ensure all elements are strings
    return separator.join(sa)


# noinspection PyShadowingBuiltins
def last(id: list[T]) -> T:
    """
    Returns the last element in the array.

    :param id: Input array
    :return: Last element in the array
    """
    if len(id) == 0:
        raise RuntimeError("Cannot get last element of an empty array!")
    return id[-1]


# noinspection PyShadowingBuiltins
def lastindexof(id: list[T], value: T) -> int:
    """
    Returns the index of the last occurrence of the specified value in the array.

    The search is tolerant, like ``indexof``.

    :param id: Input array
    :param value: Value to search for
    :return: Index of the last occurrence of the specified value in the array
    """
    # Tolerance measured on TradingView (probe m551)
    for i in builtins.range(len(id) - 1, -1, -1):
        if _equal(id[i], value):
            return i
    return -1


# noinspection PyShadowingBuiltins
def max(id: list[Number], nth: int = 0) -> Number:
    """
    Returns the maximum value in the array, or the nth largest value.

    na elements are ignored. ``nth`` is 0-based: 0 is the maximum, 1 the second
    largest, and so on. An na ``nth`` is treated as 0 and a float one is
    truncated to an integer. Returns na if the array holds no non-na values or
    ``nth`` is out of range.

    :param id: Input array
    :param nth: Rank of the maximum to return (0 = maximum)
    :return: The nth largest value in the array, or na
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100, array [10, 20, 30, 40]):
    # max(a, na) -> 40, the same as nth = 0, while nth = 1 gives 30.
    if not (nth == nth):  # is_na_arg
        nth = 0
    nth = int(nth)  # float-carried integer rank, see get()
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        return id[0] if id else NA(None)
    if nth == 0:
        return builtins.max(a)
    if nth < 0 or nth >= len(a):
        return cast(Number, NA(builtins.type(a[0])))
    return sorted(a, reverse=True)[nth]


# noinspection PyShadowingBuiltins
def median(id: list[Number]) -> float:
    """
    Returns the median value of the elements in the array.

    :param id: Input array
    :return: Median value of the elements in the array, or na if the array is empty
    """
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        return na_float
    return statistics.median(a)


# noinspection PyShadowingBuiltins
def min(id: list[Number], nth: int = 0) -> Number:
    """
    Returns the minimum value in the array, or the nth smallest value.

    na elements are ignored. ``nth`` is 0-based: 0 is the minimum, 1 the second
    smallest, and so on. An na ``nth`` is treated as 0 and a float one is
    truncated to an integer. Returns na if the array holds no non-na values or
    ``nth`` is out of range.

    :param id: Input array
    :param nth: Rank of the minimum to return (0 = minimum)
    :return: The nth smallest value in the array, or na
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100, array [10, 20, 30, 40]):
    # min(a, na) -> 10, the same as nth = 0, while nth = 1 gives 20.
    if not (nth == nth):  # is_na_arg
        nth = 0
    nth = int(nth)  # float-carried integer rank, see get()
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        return id[0] if id else NA(None)
    if nth == 0:
        return builtins.min(a)
    if nth < 0 or nth >= len(a):
        return cast(Number, NA(builtins.type(a[0])))
    return sorted(a)[nth]


# noinspection PyShadowingBuiltins
def mode(id: list[T]) -> T:
    """
    Returns the most frequently occurring element in the array.

    :param id: Input array
    :return: Most frequently occurring element in the array, or na if the array is empty
    """
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        # An all-na array still knows its element type through its na elements;
        # a truly empty one does not, so it gets a typeless na
        return id[0] if id else NA(None)
    return statistics.mode(a)


# noinspection PyShadowingNames
def _na_size(size: int | NA) -> int:
    """
    Normalize an array constructor ``size`` argument.

    An ``na`` size (e.g. ``array.new<line>(na)``) is treated as 0 and produces an
    empty array instead of failing. A genuinely negative size is still rejected.

    :param size: Requested array size, possibly ``na``
    :return: Non-negative integer size
    """
    if not (size == size):  # is_na_arg
        return 0
    assert size >= 0, "Size must be >=0!"
    return int(size)


# noinspection PyShadowingNames
def new_box(size: int | NA = 0, initial_value: Box = NA(Box)) -> list[Box]:
    """
    Creates a new array of box objects of the specified size, with each element initialized
    to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of box objects
    """
    size = _na_size(size)
    assert isinstance(initial_value, (Box, NA)), "Initial value must be Box!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_line(size: int | NA = 0, initial_value: Line = NA(Line)) -> list[Line]:
    """
    Creates a new array of line objects of the specified size, with each element initialized
    to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of line objects
    """
    size = _na_size(size)
    assert isinstance(initial_value, (Line, NA)), "Initial value must be Line!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_label(size: int | NA = 0, initial_value: Label = NA(Label)) -> list[Label]:
    """
    Creates a new array of label objects of the specified size, with each element initialized
    to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of label objects
    """
    size = _na_size(size)
    assert isinstance(initial_value, (Label, NA)), "Initial value must be Label!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_linefill(size: int | NA = 0,
                 initial_value: LineFill = NA(LineFill)) -> list[LineFill]:
    """
    Creates a new array of linefill objects of the specified size, with each element initialized
    to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of linefill objects
    """
    size = _na_size(size)
    assert isinstance(initial_value, (LineFill, NA)), "Initial value must be LineFill!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_table(size: int | NA = 0, initial_value: Table = NA(Table)) -> list[Table]:
    """
    Creates a new array of table objects of the specified size, with each element initialized
    to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of table objects
    """
    size = _na_size(size)
    assert isinstance(initial_value, (Table, NA)), "Initial value must be Table!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new(size: int | NA = 0, initial_value: T = NA(T)) -> list[T]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    size = _na_size(size)
    return [initial_value] * size


# noinspection PyShadowingNames
def new_bool(size: int | NA = 0, initial_value: bool = NA(bool)) -> list[bool]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    size = _na_size(size)
    assert isinstance(initial_value, (bool, NA)), "Initial value must be bool!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_color(size: int | NA = 0, initial_value: Color = NA(Color)) -> list[Color]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    size = _na_size(size)
    assert isinstance(initial_value, (Color, NA)), "Initial value must be Color!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_float(size: int | NA = 0, initial_value: float | int = na_float) -> list[float]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    size = _na_size(size)
    assert isinstance(initial_value, (float, int, NA)), "Initial value must be float!"
    if isinstance(initial_value, int):
        initial_value = float(initial_value)
    return [initial_value] * size


# noinspection PyShadowingNames
def new_int(size: int | NA = 0, initial_value: int = NA(int)) -> list[int]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    size = _na_size(size)
    assert isinstance(initial_value, (int, NA)), "Initial value must be int!"
    return [initial_value] * size


# noinspection PyShadowingNames
def new_string(size: int | NA = 0, initial_value: str = NA(str)) -> list[str]:
    """
    Creates a new array of the specified size, with each element initialized to the specified value.

    :param size: Size of the new array
    :param initial_value: Initial value to set for each element in the array
    :return: New array of the specified size
    """
    size = _na_size(size)
    assert isinstance(initial_value, (str, NA)), "Initial value must be str!"
    return [initial_value] * size


def _select_linear_interpolation(non_na: list[float], n: int, percentage: float) -> float:
    """
    Selection half of :func:`percentile_linear_interpolation`, shared with the
    rolling window implementation in ``ta``.

    ``non_na`` is the ascending-sorted numeric part of a conceptual array of
    ``n`` elements whose remaining ``n - len(non_na)`` na elements sort to the
    end (as if they were the largest values).

    :param non_na: Ascending-sorted numeric values
    :param n: Total conceptual array length, na elements included
    :param percentage: Percentile (0-100, not 0-1)
    :return: Interpolated value at the given percentile, or na if the percentage is na
    :raises ValueError: If percentage is not in [0, 100]
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): an na percentage yields na
    # here, unlike the nearest-rank form, which answers it as 0. The guard lives in
    # the shared selector so the array face and ``ta``'s rolling face agree.
    if not (percentage == percentage):  # is_na_arg
        return na_float
    if not (0 <= percentage <= 100):
        # Measured: TradingView halts on a percentage outside [0, 100] (RE10002),
        # so this is an error there too, not a tolerated na-like argument.
        raise ValueError("Percentage must be between 0 and 100")
    m = len(non_na)

    # 1-based interpolation position over the full length
    pos = n * percentage / 100.0 + 0.5
    # Snap to an exact integer rank when floating-point noise leaves us just shy
    nearest = round(pos)
    if builtins.abs(pos - nearest) < 1e-9:
        pos = float(nearest)

    if pos <= 1:
        return non_na[0] if m > 0 else na_float
    if pos >= n:
        return non_na[-1] if m == n else na_float

    lower = math.floor(pos)  # 1-based lower rank
    frac = pos - lower
    if frac == 0:
        return non_na[lower - 1] if lower <= m else na_float
    if m < n:
        return na_float
    # Weighted average of the two ranks, not the ``lo + frac * (hi - lo)`` form:
    # the two are algebraically equal but round differently, and TradingView
    # follows this one (probe m580, 22k bars, zero mismatches).
    return non_na[lower - 1] * (1 - frac) + non_na[lower] * frac


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentile_linear_interpolation(id: list[float], percentage: float) -> float:
    """
    Calculate the percentile value using linear interpolation.

    Values are sorted ascending with na elements pushed to the end (as if they
    were the largest values). The interpolation position is 1-based over the full
    array length, ``pos = n * percentage / 100 + 0.5``, clamped to the array
    bounds.

    Without na the value is interpolated linearly between the two ranks
    straddling ``pos``. Once the array holds any na element, only the low-end
    clamp and a ``pos`` landing exactly on an integer rank yield a value; every
    fractional position returns na, even when both neighbouring values are
    numeric. An exact rank falling in the sorted-to-end na tail likewise yields
    na. An na ``percentage`` yields na.

    An empty array yields na, and so does an na ``percentage``.

    :param id: List of numeric values, possibly containing na elements
    :param percentage: Percentile (0-100, not 0-1)
    :return: Interpolated value at the given percentile, or na (see above)
    :raises ValueError: If percentage is not in [0, 100]
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): an empty array returns na
    # and keeps running -- it is not the error an out-of-range percentage gets.
    if not id:
        return na_float

    non_na = sorted(v for v in id if v == v)  # non-na: na never equals itself
    return _select_linear_interpolation(non_na, len(id), percentage)


def _select_nearest_rank(non_na: list[float], n: int, percentage: float) -> float:
    """
    Selection half of :func:`percentile_nearest_rank`, shared with the rolling
    window implementation in ``ta``.

    ``non_na`` is the ascending-sorted numeric part of a conceptual array of
    ``n`` elements whose remaining ``n - len(non_na)`` na elements sort to the
    end (as if they were the largest values).

    :param non_na: Ascending-sorted numeric values
    :param n: Total conceptual array length, na elements included
    :param percentage: Percentile (0-100), na counts as 0
    :return: The value at the nearest rank, or na if the rank falls on a na
             element
    :raises ValueError: If percentage is not between 0 and 100
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): an na percentage answers
    # exactly what 0 answers -- the smallest value, on [4, 3, 2, 1] as well, so it
    # really is the rank and not the first element. Holds for ``ta``'s rolling face
    # too, where it tracked ta.lowest bar by bar.
    if not (percentage == percentage):  # is_na_arg
        percentage = 0
    if not (0 <= percentage <= 100):
        # Measured: TradingView halts on a percentage outside [0, 100] (RE10002).
        raise ValueError("Percentage must be between 0 and 100")
    m = len(non_na)
    if percentage == 0:
        return non_na[0] if m > 0 else na_float

    # Calculate the rank using the ceiling function as per the nearest rank method
    rank = math.ceil(percentage * n / 100)
    # Clamp rank to be within the valid range [1, n]
    rank = builtins.max(1, builtins.min(rank, n))
    # Adjust for 0-indexed array: return the (rank-1)th element
    return non_na[rank - 1] if rank <= m else na_float


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentile_nearest_rank(id: list[float], percentage: float) -> float:
    """
    Calculate the nearest rank percentile without interpolation.

    na elements are kept and sort to the end (as if they were the largest
    values), so the full array length (na included) drives the rank. A rank that
    lands on a na element yields na, an na ``percentage`` counts as 0 and an empty
    array yields na.

    :param id: List of numeric values
    :param percentage: Percentile (0-100), na counts as 0
    :return: The value at the nearest rank for the specified percentile, or na
             if that rank falls on a na element
    :raises ValueError: If percentage is not between 0 and 100
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): an empty array returns na
    # and keeps running.
    if not id:
        return na_float

    non_na = sorted(v for v in id if v == v)  # non-na: na never equals itself
    return _select_nearest_rank(non_na, len(id), percentage)


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentrank(id: list[Number], index: int) -> float:
    """
    Returns the percentile rank of the element at the specified index.
    The percentile rank is the percentage of values less than or equal to the value at index.

    na elements are ignored when counting values at or below the target, but
    still count toward the array length. If the element at ``index`` is itself
    na, the rank is na.

    An na index is treated as index 0 and a float index is truncated to an
    integer. An array too short to have a rank denominator (fewer than two
    elements) yields na.

    :param id: Input array
    :param index: Index of the element to calculate rank for
    :return: Percentile rank (0-100), or na if the element at index is na
    :raises ValueError: If the index is out of range
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): percentrank with an na
    # index returns 0 on [10, 20, 30, 40], exactly what index 0 returns; on the
    # reversed array both give 100, so it really is index 0 and not a fixed 0.
    if not (index == index):  # is_na_arg
        index = 0
    index = int(index)  # float-carried integer index, see get()

    # Measured on TradingView (FX:EURUSD 240, bar 100): an empty array returns
    # na and keeps running, both for an na index and for index 0 -- it is not
    # the out-of-bounds error an empty array gets from get/set/remove.
    if not id:
        return na_float

    if not 0 <= index < len(id):
        # Measured on TradingView: percentrank([10, 20, 30, 40], -1) halts with
        # RE10045, so an out-of-range index is an error here, unlike in
        # get/set/remove/insert which accept negative indices.
        raise ValueError("Index out of range")

    # Measured on TradingView (FX:EURUSD 240, bar 100): percentrank on a
    # one-element array returns na for index 0 and for an na index. The rank
    # formula below divides by ``len(id) - 1``, which is zero there.
    if len(id) == 1:
        return na_float

    # Get value at index
    value = id[index]
    if not (value == value):
        return na_float

    # Count non-na elements less than or equal to the target value. The
    # comparison is TOLERANT (measured on TradingView through ta.percentrank,
    # probe m548: a window whose other values sit a sub-EPSILON step above the
    # current one still ranks 100). The raw ``<=`` comes first because two
    # equal infinities have a nan difference, which the tolerance band alone
    # would reject.
    count = builtins.sum(1 for x in id
                         if x == x and (x <= value or x - value <= _EPSILON))

    # Calculate percentage
    return (count - 1) * 100 / (len(id) - 1)


# noinspection PyShadowingBuiltins
def pop(id: list[T] | SequenceView[T]) -> T:
    """
    Removes the last element from the array and returns it.

    :param id: Input array
    :return: Last element from the array
    """
    return id.pop()


# noinspection PyShadowingBuiltins
def push(id: list[T] | SequenceView[T], value: T) -> None:
    """
    Appends the specified value to the end of the array.

    :param id: Input array
    :param value: Value to append
    """
    id.append(value)


# noinspection PyShadowingBuiltins
def range(id: list[Number]) -> Number:
    """
    Returns the range of the elements in the array.

    :param id: Input array
    :return: Range of the elements in the array, or na if the array is empty
    """
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        return id[0] if id else NA(None)
    return builtins.max(a) - builtins.min(a)


# noinspection PyShadowingBuiltins
def remove(id: list[T] | SequenceView[T], index: int) -> T:
    """
    Removes the element at the specified index from the array.

    An na index removes nothing and returns na, a float index is truncated to an
    integer and a negative one addresses the array from its end, like in ``get``.

    :param id: Input array
    :param index: Index of the element to remove
    :return: The removed element, or na if the index is na
    :raises IndexError: If the index is out of range
    """
    if not (index == index):  # is_na_arg
        # Measured on TradingView (FX:EURUSD 240, bar 100):
        #   remove([10, 20, 30, 40], na) -> NaN, array unchanged
        #   remove(array.new_int(), na)  -> NaN, size 0
        # TradingView diverges on string and user-defined-type arrays, where an
        # na index is coerced to 0 and genuinely removes the head element. A
        # PyneCore array is a plain list with no runtime element type, so that
        # branch is not reproducible; the tolerant behaviour is used uniformly.
        return cast(T, _na_element(id))
    return id.pop(int(index))  # float-carried integer index, see get()


# noinspection PyShadowingBuiltins
def reverse(id: list[T] | SequenceView[T]) -> None:
    """
    Reverses the order of the elements in the array.

    :param id: Input array
    """
    id.reverse()


# noinspection PyShadowingBuiltins
def set(id: list[T] | SequenceView[T], index: int, value: T) -> None:
    """
    Sets the value of the element at the specified index in the array.

    An na index is a silent no-op, a float index is truncated to an integer and a
    negative one addresses the array from its end, like in ``get``.

    :param id: Input array
    :param index: Index of the element to set
    :param value: Value to set
    :raises IndexError: If the index is out of range
    """
    if not (index == index):  # is_na_arg
        # Measured on TradingView (FX:EURUSD 240, bar 100):
        #   set([10, 20, 30, 40], na, 99) -> 10,20,30,40 size 4
        #   set(array.new_int(), na, 9)   -> size 0
        return
    id[int(index)] = value  # float-carried integer index, see get()


# noinspection PyShadowingBuiltins
def shift(id: list[T] | SequenceView[T]) -> T:
    """
    Removes the first element from the array and returns it.

    :param id: Input array
    :return: First element from the array
    """
    return id.pop(0)


# noinspection PyShadowingBuiltins
def size(id: list[Any] | SequenceView[Any]) -> int:
    """
    Returns the number of elements in the array.

    :param id: Input array
    :return: Number of elements in the array
    """
    return len(id)


# noinspection PyShadowingBuiltins
def slice(id: list[T] | SequenceView[T], index_from: int, index_to: int) -> SequenceView[T]:
    """
    The function creates a slice from an existing array. If an object from the slice changes, the
    changes are applied to both the new and the original arrays.

    Adding to or removing from the slice changes the original array too, inside the
    slice's own bounds: pushing to a slice inserts at the slice end rather than at the
    end of the original array.

    An na ``index_from`` starts the slice at the beginning of the array and an na
    ``index_to`` ends it at the array size.

    ``index_from`` must address an existing element and ``index_to`` may reach one
    position past the last one; anything else is an error.

    :param id: Input array
    :param index_from: Index to start the sub-array from
    :param index_to: Index to end the sub-array at
    :return: Slice view of the original array
    :raises ValueError: If the indices are out of range or start after each other
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100, array [10, 20, 30, 40]):
    #   slice(a, na, 2)  -> 10,20        size 2
    #   slice(a, 1, na)  -> 20,30,40     size 3
    #   slice(a, na, na) -> 10,20,30,40  size 4
    start = int(index_from) if index_from == index_from else 0  # is_na_arg
    stop = int(index_to) if index_to == index_to else len(id)  # is_na_arg

    # Measured on TradingView (FX:EURUSD 240, array [10, 20, 30, 40]): invalid bounds
    # HALT the script, they are not normalized into a clamped or empty slice.
    #   slice(a, 0, 10) / slice(a, -1, 2) / slice(a, 0, -1) / slice(a, 4, 4) -> RE10045
    #   slice(a, 3, 1)                                                      -> RE10044
    #   slice(a, 1, 1) -> size 0, slice(a, 0, 4) -> size 4, slice(a, 3, 4) -> size 1
    # So index_from addresses an element while index_to may equal the size, and equal
    # indices are a legal empty slice. An EMPTY array is always an error, even with na
    # indices: slice(array.new<int>(0), na, na) halts with RE10045 too.
    n = len(id)
    if not 0 <= start < n:
        raise ValueError(f"Start index {start} is out of range, array size is {n}")
    if not 0 <= stop <= n:
        raise ValueError(f"End index {stop} is out of range, array size is {n}")
    if start > stop:
        raise ValueError(f"Start index {start} is greater than end index {stop}")

    return SequenceView(id)[start:stop]  # type: ignore


# noinspection PyShadowingBuiltins
def some(id: list[Any]) -> bool:
    """
    Returns true if at least one element of the id array is true, false otherwise.

    :param id: Input array
    :return: True if at least one element of the id array is true, false otherwise
    """
    return any(id)


# noinspection PyShadowingBuiltins
def _na_sorts_first(sample: Any) -> bool:
    """
    Report whether na belongs at the front of the sorted result.

    :param sample: Any non-na element of the array, None if it has none
    :return: True for a string array, False for a numeric one
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): na sorts to the END of a
    # numeric array (as the largest value) but to the FRONT of a string one.
    return isinstance(sample, str)


# noinspection PyShadowingBuiltins
def sort(id: list[int | float | str] | SequenceView[int | float | str],
         order: _order.Order = _order.ascending) -> None:
    """
    Sorts the elements in the array in ascending or descending order.

    na elements sort to the end of a numeric array and to the front of a string
    one; descending order is the ascending result reversed.

    :param id: Input array
    :param order: Order to sort the elements in
    """
    # Python's own sort cannot express this: every comparison against na is False,
    # so a single na element leaves neighbouring values in their original order --
    # sort([30, na, 10]) did not sort at all. Measured on TradingView:
    #   sort([30, 20, 10, na])            -> 10, 20, 30, na
    #   sort([30, 20, 10, na], descending) -> na, 30, 20, 10
    #   sort(["b", "a", na])              -> na, "a", "b"
    non_na: list[Any] = []
    nas: list[Any] = []
    for value in id:
        (non_na if value == value else nas).append(value)
    non_na.sort()
    ordered = nas + non_na if _na_sorts_first(non_na[0] if non_na else None) else non_na + nas
    if order == _order.descending:
        ordered.reverse()
    id[:] = ordered


# noinspection PyShadowingBuiltins
def sort_indices(id: list[T], order: _order.Order = _order.ascending) -> list[int]:
    """
    Returns an array of indices which, when used to index the original array, will access its elements
    in their sorted order. It does not modify the original array.

    The indices of na elements go where ``sort`` would put the elements
    themselves: to the end for a numeric array, to the front for a string one.

    :param id: Input array
    :param order: Order to sort the elements in
    :return: Array of indices to access the elements in their sorted order
    """
    # Measured on TradingView: sort_indices([na, na, 5, 1]) -> 3, 2, 0, 1, so the
    # na indices keep their original relative order at the end of the result.
    non_na: list[int] = []
    nas: list[int] = []
    for i, value in enumerate(id):
        (non_na if value == value else nas).append(i)
    non_na.sort(key=id.__getitem__)  # type: ignore[arg-type]
    sample = id[non_na[0]] if non_na else None
    indices = nas + non_na if _na_sorts_first(sample) else non_na + nas
    if order == _order.descending:
        indices.reverse()
    return indices


# noinspection PyShadowingBuiltins,PyShadowingNames
def standardize(id: list[float | int]) -> list[float | int]:
    """
    Standardizes the input array: every element becomes its z-score against the
    population mean and standard deviation.

    na elements are left out of both statistics and stay na in the result, so the
    population divisor is the number of numeric elements. An array whose numeric
    elements are all equal standardizes to 1.0, and one with no numeric element at
    all to na.

    :param id: A list of numeric values (int or float), possibly holding na
    :return: A new list containing the standardized values
    """
    # Measured on TradingView (FX:EURUSD 240, bar 100): an int array standardizes
    # exactly like the same values typed as float -- there is no -1/0/1
    # thresholding -- and [1, 2, 3, na] gives the z-scores of [1, 2, 3] with na in
    # the fourth slot, so the divisor is 3 and not 4.
    values = _numeric(id)
    if not values:
        return [na_float] * len(id)

    # The population mean and standard deviation ``variance``/``stdev`` measure,
    # so the three agree to the last bit (TV-verified over 29k bars on both ends
    # of a growing array).
    mean, var = _population_stats(values)
    stdev = math.sqrt(var)
    if stdev == 0:
        return [1.0 if v == v else na_float for v in id]
    return [(v - mean) / stdev if v == v else na_float for v in id]


# noinspection PyShadowingBuiltins
def stdev(id: list[Number], biased: bool = True) -> float:
    """
    Returns the standard deviation of the elements in the array.

    :param id: Input array
    :param biased: If True, calculates the biased standard deviation. If False, calculates the
                   unbiased standard deviation.
    :return: Standard deviation of the elements in the array, or na if the array is empty
    """
    # The square root of ``variance`` with the same divisor -- TV-verified on the
    # same 29k-bar probe for both the biased and the unbiased path.
    var = variance(id, biased)
    if not (var == var):
        return na_float
    return math.sqrt(var)


# noinspection PyShadowingBuiltins
def sum(id: list[float | int]) -> float | int:
    """
    Returns the sum of the elements in the array.

    :param id: Input array
    :return: Sum of the elements in the array, or na if the array is empty
    """
    a = [i for i in id if i == i]  # non-na: neither NA nor nan equals itself
    if not a:
        return na_float
    return _seq_sum(a)


# noinspection PyShadowingBuiltins
def unshift(id: list[T] | SequenceView[T], value: T) -> None:
    """
    Prepends the specified value to the beginning of the array.

    :param id: Input array
    :param value: Value to prepend
    """
    id.insert(0, value)


# noinspection PyShadowingBuiltins
def variance(id: list[Number], biased: bool = True) -> float:
    """
    Returns the variance of the elements in the array.

    :param id: Input array
    :param biased: If True, calculates the biased variance. If False, calculates the unbiased variance.
    :return: Variance of the elements in the array, or na if the array is empty
    """
    # MEASURED (BINANCE:BTCUSDT@30, 29k bars, growing arrays of both closes and
    # bar indices, every bar bit-identical): with p and q the front-to-back sums
    # of the elements and of their squares and m = p / length, TradingView
    # computes
    #   biased:   max(0, q / length - m * m)
    #   unbiased: max(0, q / (length - 1) - (p / (length - 1)) * m)
    # The unbiased division is distributed over p, not over the product, exactly
    # as written. A single numeric element is 0.0 biased and na unbiased.
    a = _numeric(id)
    if not a:
        return na_float
    length = len(a)
    if length < 2:
        return 0.0 if biased else na_float

    p, q = _moments(a)
    mean = p / length
    if biased:
        return builtins.max(0.0, q / length - mean * mean)
    return builtins.max(0.0, q / (length - 1) - (p / (length - 1)) * mean)
