from collections.abc import MutableSequence
from typing import Any, TypeVar, Iterable, Iterator, cast, overload

T = TypeVar('T')


class SequenceView(MutableSequence[T]):
    """
    A view for list slice

    Useful for creating a slice of list but modifying the slice will modify the original list.
    And vice versa.
    """

    # Measured on TradingView (FX:EURUSD 240) with array.slice(a, 0, 2) on [10, 20, 30, 40]:
    # a view is a plain index range over the parent, and every mutation of the view writes
    # through inside the view's own bounds -- push lands at the slice end (parent index 2,
    # giving 10,20,99,30,40), pop takes the slice's last element (20, not the parent's 40)
    # and clear() cuts the sliced elements out of the parent. The stored range never
    # follows the parent's content: removing the parent's first element leaves the [0, 2)
    # view showing 20,30.
    #
    # Accessing a mapped index the parent no longer has HALTS the script. Measured
    # 2026-08-18 (TradingView v6): slice(a, 3, 5) on a 5-element array, then
    # array.remove(a, 0), then array.indexof on the slice -> runtime error. An earlier
    # revision generalized the in-bounds observations above into silently clipping the
    # view to the parent's length; that swallowed the error (an indexof over the stale
    # view returned -1) and regressed the expds_160 doc sample, which exists to
    # demonstrate the halt. The stored range is therefore used RAW: a dead index reaches
    # the parent list and raises its native IndexError, matching both the measurement
    # and the pre-clip behavior ("list index out of range").

    __slots__ = ('sequence', 'range')

    # ``range_object`` is quoted because a method signature is evaluated in the CLASS
    # scope, where ``range`` is this class's own attribute, not the builtin type.
    def __init__(self, sequence: MutableSequence[T], range_object: 'range | None' = None) -> None:
        self.range: range = range_object if range_object is not None else range(len(sequence))
        self.sequence = sequence

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> 'SequenceView[T]': ...

    def __getitem__(self, key: int | slice) -> 'T | SequenceView[T]':
        r = self.range
        if isinstance(key, slice):
            return SequenceView(self.sequence, r[key])
        else:
            return self.sequence[r[key]]

    @overload
    def __setitem__(self, key: int, value: T) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, key: int | slice, value: Any) -> None:
        r = self.range
        if isinstance(key, slice):
            # Slice assignment takes one value PER position, like a list's does.
            # Storing the iterable itself in every addressed slot instead turned
            # array.fill(array.slice(a, 0, 2), 5) on [10, 20, 30, 40] into
            # [[5, 5], [5, 5], 30, 40] -- nested lists written into the parent.
            indices = r[key]
            values = list(value)
            if len(values) != len(indices):
                raise ValueError(f"Cannot assign {len(values)} values "
                                 f"to a view slice of size {len(indices)}")
            for i, item in zip(indices, values):
                self.sequence[i] = item
        else:
            self.sequence[r[key]] = value

    def __delitem__(self, key: int | slice) -> None:
        r = self.range
        if isinstance(key, slice):
            # Delete from the parent at the mapped indices, highest first so the lower
            # ones stay valid, then shrink this view by as many positions
            indices = r[key]
            for i in sorted(indices, reverse=True):
                del self.sequence[i]
            self.range = range(self.range.start, self.range.stop - len(indices))
        else:
            del self.sequence[r[key]]
            self.range = range(self.range.start, self.range.stop - 1)

    def insert(self, index: int, value: T) -> None:
        # Pine's array.push on a slice appends at the SLICE end, not at the parent's:
        # the view's own index maps to the parent index, and the view grows by one.
        # ``index`` is clamped like list.insert clamps its own.
        r = self.range
        size = len(r)
        if index < 0:
            index += size
        if index < 0:
            index = 0
        elif index > size:
            index = size
        self.sequence.insert(r.start + index, value)
        self.range = range(self.range.start, self.range.stop + 1)

    def __len__(self) -> int:
        return len(self.range)

    def __iter__(self) -> Iterator[T]:
        r = self.range
        for i in r:
            yield self.sequence[i]

    def __repr__(self) -> str:
        return f"SequenceView({self.sequence!r}, {self.range!r})"

    def __str__(self) -> str:
        # A view over a view (array.slice of a slice) must render its own elements,
        # not the nested view object, so walk down to the concrete container first.
        seq: Any = self.sequence
        while isinstance(seq, SequenceView):
            seq = seq.sequence
        if isinstance(seq, str):
            return ''.join(cast('Iterator[str]', self))
        elif isinstance(seq, (list, tuple)):
            return str(type(seq)(self))
        else:
            return repr(self)
