from __future__ import annotations
from typing import TypeVar, Generic, MutableSequence, Iterator, cast

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

    def __getitem__(self, key: int | slice) -> T | SequenceView[T]:
        if isinstance(key, slice):
            return SequenceView(self.sequence, self.range[key])
        else:
            return self.sequence[self.range[key]]

    def __setitem__(self, key: int | slice, value: T) -> None:
        if isinstance(key, slice):
            for i in self.range[key]:
                self.sequence[i] = value
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
