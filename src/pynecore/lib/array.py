from typing import TypeVar, Any, cast

import builtins

import math
import statistics

from ..utils.sequence_view import SequenceView

from ..types.na import NA
from ..types.color import Color
from ..types.box import Box
from ..types.line import Line
from ..types.label import Label
from ..types.linefill import LineFill
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


# noinspection PyShadowingBuiltins
def _non_na(id: list[T]) -> list[T]:
    """
    Return the array's elements with na values removed.

    TradingView's array math and statistics reductions ignore na elements and
    reduce over the remaining values only, yielding na when none remain. Every
    such reduction filters its input through this helper so their na handling
    stays consistent with Pine.

    :param id: Input array, possibly containing na elements
    :return: New list containing only the non-na elements, in original order
    """
    return [i for i in id if not isinstance(i, NA)]


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
    a = _non_na(id)
    if not a:
        return NA(float)
    return builtins.sum(a) / len(a)


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
    Returns the index of the specified value in the sorted array using binary search.
    If the value is not found, returns the index of the leftmost element greater than the value.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the specified value in the sorted array, or the index of the leftmost element
             greater than the value
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
    return low - 1


# noinspection PyShadowingBuiltins
def binary_search_rightmost(id: list[Any], val: Any) -> int:
    """
    Returns the index of the specified value in the sorted array using binary search.
    If the value is not found, returns the index of the rightmost element less than the value.
    The array to search must be sorted in ascending order.

    :param id: Input array
    :param val: Value to search for
    :return: Index of the specified value in the sorted array, or the index of the rightmost element less than the value
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
    return high + 1


# noinspection PyShadowingBuiltins
def clear(id: list[Any]) -> None:
    """
    Removes all elements from the array.

    :param id: Input array
    """
    id.clear()


# noinspection PyShadowingBuiltins
def concat(id1: list[T], id2: list[T]) -> list[T]:
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
    pairs = [(v1, v2) for v1, v2 in zip(id1, id2)
             if not isinstance(v1, NA) and not isinstance(v2, NA)]
    if not pairs:
        return NA(float)
    # Online (Welford) co-moment — matches TradingView bit-for-bit for both
    # the biased and unbiased result, where the classic two-pass
    # ``sum((x-mx)*(y-my)) / divisor`` lands a couple of ulps off on the
    # unbiased path (TV-verified in test_003_array_functions_float).
    length = 0
    mean1 = 0.0
    mean2 = 0.0
    comoment = 0.0
    for v1, v2 in pairs:
        length += 1
        d1 = v1 - mean1
        mean1 += d1 / length
        mean2 += (v2 - mean2) / length
        comoment += d1 * (v2 - mean2)
    divisor = (length - 1) if not biased else length
    if divisor == 0:
        return 0.0
    return comoment / divisor


# noinspection PyShadowingBuiltins
def every(id: list[Any]) -> bool:
    """
    Returns true if all elements of the id array are true, false otherwise.

    :param id: Input array
    :return: True if all elements of the id array are true, false otherwise
    """
    return all(id)


# noinspection PyShadowingBuiltins
def fill(id: list[T], value: T, index_from: int = 0, index_to: int | NA = NA(int)) -> None:
    """
    Fills the elements in the array with the specified value.

    :param id: Input array
    :param value: Value to fill
    :param index_from: Index to start filling from
    :param index_to: Index to stop filling at
    """
    if isinstance(index_to, NA):
        index_to = len(id)
    id[index_from:index_to] = [value] * (index_to - index_from)


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

    :param id: Input array
    :param index: Index of the element to return
    :return: Element at the specified index in the array
    """
    return id[index]


# noinspection PyShadowingBuiltins
def includes(id: list[T], value: T) -> bool:
    """
    Returns true if the array contains the specified value, false otherwise.

    :param id: Input array
    :param value: Value to search for
    :return: True if the array contains the specified value, false otherwise
    """
    return value in id


# noinspection PyShadowingBuiltins
def indexof(id: list[T], value: T) -> int:
    """
    Returns the index of the first occurrence of the specified value in the array.

    :param id: Input array
    :param value: Value to search for
    :return: Index of the first occurrence of the specified value in the array
    """
    try:
        return id.index(value)
    except ValueError:
        return -1


# noinspection PyShadowingBuiltins
def insert(id: list[T], index: int, value: T) -> None:
    """
    Inserts the specified value at the specified index in the array.

    :param id: Input array
    :param index: Index to insert the value at
    :param value: Value to insert
    """
    id.insert(index, value)


# noinspection PyShadowingBuiltins
def join(id: list, separator: str) -> str:
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

    :param id: Input array
    :param value: Value to search for
    :return: Index of the last occurrence of the specified value in the array
    """
    try:
        return len(id) - 1 - id[::-1].index(value)
    except ValueError:
        return -1


# noinspection PyShadowingBuiltins
def max(id: list[Number], nth: int = 0) -> Number:
    """
    Returns the maximum value in the array, or the nth largest value.

    na elements are ignored. ``nth`` is 0-based: 0 is the maximum, 1 the second
    largest, and so on. Returns na if the array holds no non-na values or ``nth``
    is out of range.

    :param id: Input array
    :param nth: Rank of the maximum to return (0 = maximum)
    :return: The nth largest value in the array, or na
    """
    a = _non_na(id)
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
    a = _non_na(id)
    if not a:
        return NA(float)
    return statistics.median(a)


# noinspection PyShadowingBuiltins
def min(id: list[Number], nth: int = 0) -> Number:
    """
    Returns the minimum value in the array, or the nth smallest value.

    na elements are ignored. ``nth`` is 0-based: 0 is the minimum, 1 the second
    smallest, and so on. Returns na if the array holds no non-na values or ``nth``
    is out of range.

    :param id: Input array
    :param nth: Rank of the minimum to return (0 = minimum)
    :return: The nth smallest value in the array, or na
    """
    a = _non_na(id)
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
    a = _non_na(id)
    if not a:
        # An all-na array still knows its element type through its na elements;
        # a truly empty one does not, so it gets a typeless na
        return id[0] if id else NA(None)
    return statistics.mode(a)


# noinspection PyShadowingNames
def _na_size(size: int | NA) -> int:
    """
    Normalize an array constructor ``size`` argument.

    TradingView treats an ``na`` size (e.g. ``array.new<line>(na)``) as 0,
    producing an empty array, so mirror that instead of failing. A genuinely
    negative size is still rejected.

    :param size: Requested array size, possibly ``na``
    :return: Non-negative integer size
    """
    if isinstance(size, NA):
        return 0
    assert size >= 0, "Size must be >=0!"
    return size


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
def new_float(size: int | NA = 0, initial_value: float | int = NA(float)) -> list[float]:
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


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentile_linear_interpolation(id: list[float], percentage: float) -> float:
    """
    Calculate the percentile value using linear interpolation, following TradingView's logic.

    Values are sorted ascending with na elements pushed to the end (as if they
    were the largest values). The interpolation position is 1-based over the full
    array length, ``pos = n * percentage / 100 + 0.5``, clamped to the array
    bounds.

    Without na the value is interpolated linearly between the two ranks
    straddling ``pos``. TradingView diverges once the array holds any na element:
    it then yields a value only for the low-end clamp or for a ``pos`` that lands
    exactly on an integer rank, and returns na for every fractional position --
    even when both neighbouring values are numeric. An exact rank falling in the
    sorted-to-end na tail likewise yields na.

    :param id: List of numeric values, possibly containing na elements
    :param percentage: Percentile (0-100, not 0-1)
    :return: Interpolated value at the given percentile, or na (see above)
    :raises ValueError: If arr is empty or percentage is not in [0, 100]
    """
    if not id:
        raise ValueError("Input array is empty")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    has_na = any(isinstance(v, NA) for v in id)
    # filter() instead of a comprehension: PyCharm mis-narrows `not isinstance`
    # inside comprehension conditions (elements would type as NA, not float)
    non_na = sorted(filter(lambda v: not isinstance(v, NA), id))
    sorted_arr = non_na + [NA(float)] * (len(id) - len(non_na))
    n = len(id)

    # 1-based interpolation position over the full length
    pos = n * percentage / 100.0 + 0.5
    # Snap to an exact integer rank when floating-point noise leaves us just shy
    nearest = round(pos)
    if builtins.abs(pos - nearest) < 1e-9:
        pos = float(nearest)

    if pos <= 1:
        return sorted_arr[0]
    if pos >= n:
        return sorted_arr[-1]

    lower = math.floor(pos)  # 1-based lower rank
    frac = pos - lower
    if frac == 0:
        return sorted_arr[lower - 1]
    if has_na:
        return NA(float)
    return sorted_arr[lower - 1] + frac * (sorted_arr[lower] - sorted_arr[lower - 1])


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentile_nearest_rank(id: list[float], percentage: float) -> float:
    """
    Calculate the nearest rank percentile without interpolation.

    Matches TradingView: na elements are kept and sort to the end (as if they
    were the largest values), so the full array length (na included) drives the
    rank. A rank that lands on a na element yields na.

    :param id: List of numeric values
    :param percentage: Percentile (0-100)
    :return: The value at the nearest rank for the specified percentile, or na
             if that rank falls on a na element
    :raises ValueError: If arr is empty or percentage is not between 0 and 100
    """
    if not id:
        raise ValueError("Input array is empty")
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    # filter() instead of a comprehension: see percentile_linear_interpolation
    non_na = sorted(filter(lambda v: not isinstance(v, NA), id))
    sorted_arr = non_na + [NA(float)] * (len(id) - len(non_na))
    n = len(id)
    if percentage == 0:
        return sorted_arr[0]

    # Calculate the rank using the ceiling function as per the nearest rank method
    rank = math.ceil(percentage * n / 100)
    # Clamp rank to be within the valid range [1, n]
    rank = builtins.max(1, builtins.min(rank, n))
    # Adjust for 0-indexed array: return the (rank-1)th element
    return sorted_arr[rank - 1]


# noinspection PyShadowingBuiltins,PyShadowingNames
def percentrank(id: list[Number], index: int) -> float:
    """
    Returns the percentile rank of the element at the specified index.
    The percentile rank is the percentage of values less than or equal to the value at index.

    Matches TradingView: na elements are ignored when counting values at or below
    the target, but still count toward the array length. If the element at
    ``index`` is itself na, the rank is na.

    :param id: Input array
    :param index: Index of the element to calculate rank for
    :return: Percentile rank (0-100), or na if the element at index is na
    :raises ValueError: If input array is empty or index is out of range
    """
    if not id:
        raise ValueError("Input array is empty")

    if not 0 <= index < len(id):
        raise ValueError("Index out of range")

    # Get value at index
    value = id[index]
    if isinstance(value, NA):
        return NA(float)

    # Count non-na elements less than or equal to the target value
    count = builtins.sum(1 for x in id if not isinstance(x, NA) and x <= value)

    # Calculate percentage
    return (count - 1) * 100 / (len(id) - 1)


# noinspection PyShadowingBuiltins
def pop(id: list[T]) -> T:
    """
    Removes the last element from the array and returns it.

    :param id: Input array
    :return: Last element from the array
    """
    return id.pop()


# noinspection PyShadowingBuiltins
def push(id: list[T], value: T) -> None:
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
    a = _non_na(id)
    if not a:
        return id[0] if id else NA(None)
    return builtins.max(a) - builtins.min(a)


# noinspection PyShadowingBuiltins
def remove(id: list[T], index: int) -> T:
    """
    Removes the element at the specified index from the array.

    :param id: Input array
    :param index: Index of the element to remove
    :return: The removed element
    """
    return id.pop(index)


# noinspection PyShadowingBuiltins
def reverse(id: list[T]) -> None:
    """
    Reverses the order of the elements in the array.

    :param id: Input array
    """
    id.reverse()


# noinspection PyShadowingBuiltins
def set(id: list[T] | SequenceView[T], index: int, value: T) -> None:
    """
    Sets the value of the element at the specified index in the array.

    :param id: Input array
    :param index: Index of the element to set
    :param value: Value to set
    """
    id[index] = value


# noinspection PyShadowingBuiltins
def shift(id: list[T]) -> T:
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
def slice(id: list[T], index_from: int, index_to: int) -> SequenceView[T]:
    """
    The function creates a slice from an existing array. If an object from the slice changes, the
    changes are applied to both the new and the original arrays.

    :param id: Input array
    :param index_from: Index to start the sub-array from
    :param index_to: Index to end the sub-array at
    :return: Slice view of the original array
    """
    return SequenceView(id)[int(index_from):int(index_to)]  # type: ignore


# noinspection PyShadowingBuiltins
def some(id: list[Any]) -> bool:
    """
    Returns true if at least one element of the id array is true, false otherwise.

    :param id: Input array
    :return: True if at least one element of the id array is true, false otherwise
    """
    return any(id)


# noinspection PyShadowingBuiltins
def sort(id: list[int | float | str], order: _order.Order = _order.ascending) -> None:
    """
    Sorts the elements in the array in ascending or descending order.

    :param id: Input array
    :param order: Order to sort the elements in
    """
    id.sort(reverse=order == _order.descending)


# noinspection PyShadowingBuiltins
def sort_indices(id: list[T], order: _order.Order = _order.ascending) -> list[int]:
    """
    Returns an array of indices which, when used to index the original array, will access its elements
    in their sorted order. It does not modify the original array.

    :param id: Input array
    :param order: Order to sort the elements in
    :return: Array of indices to access the elements in their sorted order
    """
    indices: list[int] = sorted(builtins.range(len(id)), key=id.__getitem__)  # type: ignore
    if order == _order.descending:
        indices.reverse()
    return indices


# noinspection PyShadowingBuiltins,PyShadowingNames
def standardize(id: list[float | int]) -> list[float | int]:
    """
    Standardizes the input array in a Pine Script-like manner:
      1) Uses a left-to-right summation for the mean (population mean).
      2) Uses a second pass for summing squared differences (population variance).
      3) Computes the population standard deviation (divisor = N).
      4) Returns the z-score for each element.
         - If all input elements are integers, it applies thresholding:
             z < -1 -> -1,
             z > 1  -> 1,
             otherwise 0
         - If any element is float, the result is the continuous z-score value.
    This version is bit-by-bit compatible with Pine Script's `standardize()` function.

    :param id: A list of numeric values (int or float).
    :return: A list containing the standardized values.
    """
    n = len(id)
    if n == 0:
        # You can decide how you want to handle the empty list.
        return []

    mean = statistics.mean(id)
    stdev = math.sqrt(statistics.mean([(v - mean) ** 2 for v in id]))
    if stdev == 0:
        # All elements are equal: TV's standardize() yields 1.0 for every element.
        z_scores = [1.0 for _ in id]
    else:
        z_scores = [(v - mean) / stdev for v in id]

    # If all values are integers, apply the thresholding to get -1, 0, or 1.
    if all(isinstance(v, int) for v in id):
        # Pine Script-style integer thresholding
        return [
            -1 if z < -1 else
            1 if z > 1 else
            0
            for z in z_scores
        ]

    # Otherwise, return the continuous z-score values.
    return z_scores


# noinspection PyShadowingBuiltins
def stdev(id: list[Number], biased: bool = True) -> float:
    """
    Returns the standard deviation of the elements in the array.

    :param id: Input array
    :param biased: If True, calculates the biased standard deviation. If False, calculates the
                   unbiased standard deviation.
    :return: Standard deviation of the elements in the array, or na if the array is empty
    """
    a = _non_na(id)
    if not a:
        return NA(float)
    if len(a) < 2:
        return 0.0
    if not biased:
        return statistics.stdev(a)
    mean = statistics.mean(a)
    return math.sqrt(statistics.mean([(v - mean) ** 2 for v in a]))


# noinspection PyShadowingBuiltins
def sum(id: list[float | int]) -> float | int:
    """
    Returns the sum of the elements in the array.

    :param id: Input array
    :return: Sum of the elements in the array, or na if the array is empty
    """
    a = _non_na(id)
    if not a:
        return NA(float)
    return builtins.sum(a)


# noinspection PyShadowingBuiltins
def unshift(id: list[T], value: T) -> None:
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
    a = _non_na(id)
    if not a:
        return NA(float)
    if len(a) < 2:
        return 0.0
    if not biased:
        return statistics.variance(a)

    length = len(a)
    mean = statistics.mean(a)
    summ = 0.0
    for v in a:
        summ += (v - mean) ** 2
    return summ / length
