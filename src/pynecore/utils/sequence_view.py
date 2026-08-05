from __future__ import annotations
from typing import Any, TypeVar, Generic, Iterable, MutableSequence, Iterator, cast, overload

T = TypeVar('T')


class SequenceView(Generic[T]):
    """
    A view for list slice

    Useful for creating a slice of list but modifying the slice will modify the original list.
    And vice versa.
    """

    __slots__ = ('sequence', 'range')

    def __init__(self, sequence: MutableSequence[T], range_object: range | None = None) -> None:
        self.range: range = range_object if range_object is not None else range(len(sequence))
        self.sequence = sequence

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> SequenceView[T]: ...

    def __getitem__(self, key: int | slice) -> T | SequenceView[T]:
        if isinstance(key, slice):
            return SequenceView(self.sequence, self.range[key])
        else:
            return self.sequence[self.range[key]]

    @overload
    def __setitem__(self, key: int, value: T) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, key: int | slice, value: Any) -> None:
        if isinstance(key, slice):
            # Slice assignment takes one value PER position, like a list's does.
            # Storing the iterable itself in every addressed slot instead turned
            # array.fill(array.slice(a, 0, 2), 5) on [10, 20, 30, 40] into
            # [[5, 5], [5, 5], 30, 40] -- nested lists written into the parent.
            indices = self.range[key]
            values = list(value)
            if len(values) != len(indices):
                raise ValueError(f"Cannot assign {len(values)} values "
                                 f"to a view slice of size {len(indices)}")
            for i, item in zip(indices, values):
                self.sequence[i] = item
        else:
            self.sequence[self.range[key]] = value

    def __len__(self) -> int:
        return len(self.range)

    def __iter__(self) -> Iterator[T]:
        for i in self.range:
            yield self.sequence[i]

    def __repr__(self) -> str:
        return f"SequenceView({self.sequence!r}, {self.range!r})"

    def __str__(self) -> str:
        if isinstance(self.sequence, str):
            return ''.join(cast('Iterator[str]', self))
        elif isinstance(self.sequence, (list, tuple)):
            return str(type(self.sequence)(self))
        else:
            return repr(self)
