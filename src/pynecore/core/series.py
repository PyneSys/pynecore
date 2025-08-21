from __future__ import annotations
from typing import TypeVar, Generic, Iterator, cast

from types import ModuleType

# noinspection PyProtectedMember
from ..types.na import NA

__all__ = ['SeriesImpl', 'inline_series']

T = TypeVar('T')


class SeriesImpl(Generic[T]):
    """
    A dynamic circular buffer that behaves similarly to Pine Script's "Series".
    Every docstring and comment is in English, as requested.

    Note: max_bars_back means the maximum number of older bars that can be indexed.
    Therefore, the actual buffer capacity is (max_bars_back + 1).
    """

    __slots__ = ('_buffer',
                 '_max_bars_back', '_max_bars_back_set',
                 '_capacity', '_write_pos', '_size',
                 '_last_bar_index')

    DEFAULT_MAX_BARS_BACK = 500  # This can be set globally by indicator or strategy commands
    MAXIMUM_MAX_BARS_BACK = 5000  # This is the maximum allowed value

    _lib: ModuleType = None  # Placeholder for the lib module

    # noinspection PyMissingConstructor
    def __init__(self, max_bars_back: int | None = None):
        """
        :param max_bars_back: Optional initial capacity for historical lookback.
                              If not provided, DEFAULT_MAX_BARS_BACK is used.
                              The actual buffer capacity will be (max_bars_back + 1).
        """
        # Importing the lib module here to avoid circular imports
        if not SeriesImpl._lib:
            from .. import lib
            SeriesImpl._lib = lib

        self._max_bars_back = max_bars_back or self.DEFAULT_MAX_BARS_BACK
        self._max_bars_back_set = max_bars_back or 0
        # Internal capacity is max_bars_back + 1 (for current candle + historical bars)
        self._capacity = self._max_bars_back + 1

        # Pre-allocated buffer with None values for better performance
        self._buffer: list[T | NA | None] = [None] * self._capacity

        # The next logical write position in a circular manner.
        self._write_pos = 0

        # The current number of valid elements in the buffer (<= self._capacity).
        self._size = 0

        # The last bar index that was accessed
        self._last_bar_index = -1

    @property
    def max_bars_back(self) -> int:
        """
        Returns the current max_bars_back, i.e. how many historical bars can be indexed.
        """
        return self._max_bars_back

    @max_bars_back.setter
    def max_bars_back(self, new_max_bars_back: int) -> None:
        """
        Resizes the circular buffer capacity to (new_value + 1).
        - If increased, old items are preserved (up to the old _size), and
          the buffer is linearized.
        - If decreased, only the most recent 'new_value + 1' items are kept.

        :param new_max_bars_back: The new 'max_bars_back' value.
        :raises ValueError: If new_value <= 0.
        """
        if new_max_bars_back <= 0:
            raise ValueError("The max_bars_back must be a positive integer!")
        if new_max_bars_back > SeriesImpl.MAXIMUM_MAX_BARS_BACK:
            raise ValueError(f"The max_bars_back cannot exceed {self.MAXIMUM_MAX_BARS_BACK}!")

        if new_max_bars_back == self._max_bars_back:
            return  # No change

        old_size = self._size
        old_write_pos = self._write_pos

        new_capacity = new_max_bars_back + 1
        if new_capacity > old_size == old_write_pos:
            # The buffer is not yet full, and the write position is at the end.
            # We can just modify the max_bars_back and capacity, and extend the buffer
            self._max_bars_back = new_max_bars_back
            self._max_bars_back_set = new_max_bars_back
            self._capacity = new_capacity
            # Extend pre-allocated buffer with None values
            self._buffer.extend([None] * (new_capacity - len(self._buffer)))
            return

        old_buffer = self._buffer
        old_capacity = self._capacity

        # Number of items to keep: either all old items or new_capacity, whichever is smaller.
        items_to_keep = min(old_size, new_capacity)

        # Build a new pre-allocated buffer.
        new_buffer: list[T | NA[T] | None] = [None] * new_capacity

        if items_to_keep > 0 and old_buffer:
            # The newest item is at (old_write_pos - 1) in a circular manner.
            # We'll copy the most recent 'items_to_keep' from the old buffer in linear order.
            start_idx = (old_write_pos - items_to_keep) % old_capacity

            for i in range(items_to_keep):
                src_idx = (start_idx + i) % old_capacity
                new_buffer[i] = old_buffer[src_idx]

        # Update references
        self._buffer = new_buffer
        self._max_bars_back = new_max_bars_back
        self._max_bars_back_set = new_max_bars_back
        self._capacity = new_capacity
        self._size = items_to_keep
        self._write_pos = items_to_keep  # The next free slot is after the last copied item

    def add(self, value: T | NA[T]) -> T | NA[T]:
        """
        Adds a new candle (data point) to the buffer.
        - If the buffer is not yet at full capacity (size < capacity), it appends the new value.
        - Otherwise, it overwrites the oldest data in a circular manner.
        :param value: The new data to be added.
        :return: The same value that was added (for chaining or inline usage).
        """
        # Optimize attribute lookup by using local variable
        lib = SeriesImpl._lib

        # Set data instead of adding a new one if the bar index is the same
        if self._last_bar_index == lib.bar_index:
            return self.set(value)

        if self._size < self._capacity:
            # The buffer is not yet full - use direct indexing to pre-allocated buffer
            self._buffer[self._write_pos] = value
            self._size += 1
            self._write_pos += 1
        else:
            # The buffer is full: overwrite in circular fashion
            pos = self._write_pos
            if pos >= self._capacity:
                pos = 0
                self._write_pos = 1
            else:
                self._write_pos += 1
            self._buffer[pos] = value

        # Store the last bar index to prevent adding more than one value per bar
        self._last_bar_index = lib.bar_index

        return value

    def set(self, value: T | NA[T]) -> T | NA[T]:
        """
        Overwrites the most recently added (current) candle value.
        If there is no data yet, returns na.
        :param value: The new value for the current candle.
        :return: The value that was set, or na if buffer is empty.
        """
        if self._size == 0:
            return cast(NA[T], NA())

        pos = self._write_pos - 1
        if pos < 0:
            pos += self._capacity
        self._buffer[pos] = value
        return value

    def __getitem__(self, key: int | slice) -> T | NA[T] | ReadOnlySeriesView[T]:
        """
        Get item(s) using Pine indexing with slice support.

        :param key: Integer index or slice
        :return: Single value for integer index, ReadOnlySeriesView for slice
        :raises IndexError: If index is out of range or negative
        :raises TypeError: If key is not int or slice
        """
        if isinstance(key, float):
            key = int(key)

        if isinstance(key, int):
            # Original integer indexing behavior
            if key < 0:
                raise IndexError("Negative indices not supported!")
            if key >= self._size:
                return cast(NA[T], NA())
            pos = self._write_pos - 1 - key
            if pos < 0:
                pos += self._capacity
            result = self._buffer[pos]
            return cast(T | NA[T], result)

        elif isinstance(key, slice):
            # Handle slice notation
            start = 0 if key.start is None else key.start
            stop = self._size if key.stop is None else key.stop

            if start < 0 or stop < 0:
                raise IndexError("Negative indices not supported in slice!")

            if key.step is not None and key.step != 1:
                raise ValueError("Step value not supported in slice!")

            if stop > self._size:
                raise IndexError("Slice stop index out of range!")

            # Ensure start <= stop
            if start > stop:
                start = stop

            return ReadOnlySeriesView[T](
                cast(list[T | NA[T] | None], self._buffer),
                self._capacity,
                self._write_pos,
                self._size,
                start,
                stop
            )

        raise TypeError("Series indices must be integers or slices")  # noqa

    def __len__(self) -> int:
        """
        Returns the number of valid data points in the buffer (<= capacity).
        """
        return self._size


class ReadOnlySeriesView(Generic[T]):
    """
    A read-only view for a circular buffer slice that follows Pine-like indexing.
    Supports positive slice notation [start:stop] where start and stop refer to
    bar indices from the most recent bar.
    """
    __slots__ = ('_buffer', '_capacity', '_write_pos', '_size', '_start', '_stop')

    def __init__(self, buffer: list[T | NA[T]], capacity: int, write_pos: int, size: int,
                 start: int, stop: int) -> None:
        """
        Initialize the view.

        :param buffer: The source buffer
        :param capacity: The capacity of the buffer
        :param write_pos: The current write position
        :param size: The number of valid items
        :param start: Start index (inclusive)
        :param stop: Stop index (exclusive)
        """
        self._buffer = buffer
        self._capacity = capacity
        self._write_pos = write_pos
        self._size = size
        self._start = start
        self._stop = min(stop, size)

    def __getitem__(self, idx: int) -> T | NA[T]:
        """Get item using Pine indexing"""
        if not isinstance(idx, int):
            raise TypeError("Only integer indexing is supported")  # noqa
        if idx < 0:
            raise IndexError("Negative indices not supported")
        if idx >= len(self):
            raise IndexError("Index out of range")

        actual_idx = idx + self._start
        pos = self._write_pos - 1 - actual_idx
        if pos < 0:
            pos += self._capacity
        return self._buffer[pos]

    def __len__(self) -> int:
        """Get length of the view"""
        return self._stop - self._start

    def __iter__(self) -> Iterator[T | NA[T]]:
        """Iterate through items from newest to oldest"""
        for i in range(self._start, self._stop):
            pos = self._write_pos - 1 - i
            if pos < 0:
                pos += self._capacity
            yield self._buffer[pos]

    def __repr__(self) -> str:
        """Get string representation"""
        return f"ReadOnlySeriesView({list(self)})"


__series_function_vars__ = {'create_series': ['__series_create_series_series__']}
__series_create_series_series__: SeriesImpl = SeriesImpl()


def inline_series(value: T, idx: int) -> SeriesImpl[T]:
    """
    Inline series creation
    It is mainly for compiled codes

    :param value: The value to store
    :param idx: The index, 0 is the last, 1 is the second last and so on
    :return:
    """
    if isinstance(idx, float):
        idx = int(idx)
    global __series_create_series_series__
    __series_create_series_series__.add(value)
    result = __series_create_series_series__[idx]
    return cast(SeriesImpl[T], result)
