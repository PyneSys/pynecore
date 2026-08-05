from dataclasses import dataclass, replace
from typing import TypeVar, Type, Any, Callable, dataclass_transform

from ..lib import array, box, label, line, matrix
from ..lib import map as map_lib
from ..utils.sequence_view import SequenceView
from ..types import box as box_types
from ..types import label as label_types
from ..types import line as line_types
from ..types import linefill as linefill_types
from ..types import matrix as matrix_types
from ..types import polyline as polyline_types
from ..types import table as table_types
from ..types.na import NA

__all__ = ['udt', 'udt_copy']

T = TypeVar('T')

# Copying a builtin Pine object is not a field-wise clone. ``label``/``line``/``box``
# own a module registry: their ``copy()`` assigns a fresh vid, inserts the clone and
# enforces ``max_*_count``. Measured on TradingView, the method form ``l.copy()`` and
# the function form ``label.copy(l)`` are indistinguishable — both append to
# ``label.all`` and are evicted on equal footing with ``label.new`` (12 objects
# created over 6 bars with max_labels_count=5 left the newest 6 alive, in strict
# interleaved creation order). ``dataclasses.replace`` produces an object carrying a
# DUPLICATE vid (``vid`` is a normal init=True field) that is in no registry, never
# reaches the chart, never counts against the cap and raises nothing. A duplicate vid
# additionally aliases the original in the viz stream, where a linefill serializes its
# lines by ``vid``.
#
# ``list``/``SequenceView``/``dict``/``Matrix`` are here for the mirrored reason: they
# are not dataclasses at all, so ``replace`` raised TypeError on them.
#
# ``chart.point`` is deliberately absent: it owns no registry, so ``replace`` is
# already the correct copy for it, and only the fallback preserves ``**changes``.
#
# This table is NOT a narrower duplicate of ``pine_method._get_builtin_method``,
# which resolves ANY method name on a receiver. This one answers a different
# question — on which types copying is defined at all — and the answer is measured:
# ``line``, ``label``, ``box``, ``matrix``, ``array`` and ``map`` receivers compile
# on TradingView while ``linefill``, ``polyline`` and ``table`` are rejected, in the
# method and the namespace-function form alike. Delegating would not even reach the
# same answer: those three namespaces define no ``copy`` to look up.
_BUILTIN_COPY: dict[type, Callable[[Any], Any]] = {
    label_types.Label: label.copy,
    line_types.Line: line.copy,
    box_types.Box: box.copy,
    matrix_types.Matrix: matrix.copy,
    list: array.copy,
    SequenceView: array.copy,
    dict: map_lib.copy,
}

# Drawings Pine cannot copy. The compiler rejects a receiver it can type, but a
# container element it cannot (``arr.get(0).copy()`` on an ``array<linefill>``)
# arrives here. They are dataclasses carrying a vid, so the fallback below would
# clone them into no registry and return a drawing that never reaches the chart.
_UNCOPYABLE: frozenset[type] = frozenset({
    linefill_types.LineFill,
    polyline_types.Polyline,
    table_types.Table,
})


def udt_copy(obj: Any, **changes: Any) -> Any:
    """
    Copy a Pine value the way its own type defines copying.

    Builtin objects are dispatched to their namespace ``copy()``; every other
    object is copied field by field with the given field overrides applied.
    Copying ``na`` yields ``na``.

    :param obj: The object to copy
    :param changes: Field values to override on the copy; user-defined types only
    :return: An independent copy of ``obj``, or ``obj`` itself when it is na
    :raises TypeError: If the object's type defines no copy, or if field overrides
                       are passed for anything but a user-defined type
    """
    obj_type = type(obj)
    if obj_type in _UNCOPYABLE:
        raise TypeError(f"{obj_type.__name__} has no copy method")
    builtin = _BUILTIN_COPY.get(obj_type)
    if builtin is not None:
        if changes:
            # Not expressible in Pine: the builtin ``.copy()`` takes no arguments, and
            # only ``@udt``'s own ``copy`` forwards changes, which lands on the fallback
            # below. Refuse rather than poke attributes on a registered chart object.
            raise TypeError(f"{obj_type.__name__} copy takes no field overrides")
        return builtin(obj)
    if isinstance(obj, NA):
        if changes:
            # An na holds no fields to override. Dropping them silently would answer a
            # typo in hand-written Pyne with a plausible-looking na, so it is refused
            # on the same footing as a builtin receiver.
            raise TypeError("na copy takes no field overrides")
        # Measured on TradingView: copying an na drawing leaves the registry
        # untouched and returns na. NA is interned per declared type, so this
        # value already IS the na the namespace copy would return.
        return obj
    return replace(obj, **changes)


@dataclass_transform()
def udt(cls: Type[T]) -> Type[T]:
    """
    Custom dataclass decorator that adds a `copy` method to the class.

    This decorator applies the standard dataclass decorator and then adds
    a `copy` method that creates a copy of the instance using dataclass.replace().

    :param cls: The class to decorate
    :return: The decorated class with added copy method
    """
    # Apply the standard dataclass decorator with slots=True for better performance
    decorated_cls = dataclass(cls, slots=True)  # type: ignore

    def copy(self: T, **changes: Any) -> T:
        """
        Create a copy of this instance with optional field modifications.

        :param self: The instance to copy
        :param changes: Optional keyword arguments to override field values
        :return: A new instance with the specified changes
        """
        return udt_copy(self, **changes)

    # noinspection PyShadowingNames
    @classmethod  # noqa
    def new(cls, *args, **kwargs) -> T:
        """
        Pine Script-style constructor method.

        Creates a new instance of the class using the same arguments as __init__.
        This provides Pine Script compatibility where UDTs are created with .new().

        :param cls: The class to construct
        :param args: Positional arguments for the constructor
        :param kwargs: Keyword arguments for the constructor
        :return: A new instance of the class
        """
        return cls(*args, **kwargs)

    # Add the methods to the class
    decorated_cls.copy = copy  # type: ignore
    decorated_cls.new = new

    return decorated_cls
