from dataclasses import dataclass, replace
from typing import TypeVar, Type, Any, Callable, dataclass_transform

from ..lib import array, box, label, line, matrix
from ..lib import map as map_lib
from ..utils.sequence_view import SequenceView
from ..types import box as box_types
from ..types import label as label_types
from ..types import line as line_types
from ..types import matrix as matrix_types
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
# ``linefill``/``polyline``/``table`` are absent because Pine has no copy for them:
# TradingView rejects ``lf.copy()``, ``linefill.copy(lf)``, ``pl.copy()``,
# ``polyline.copy(pl)`` and ``t.copy()`` at compile time with CE10271, so no valid
# Pine reaches them. They do carry a vid, so the fallback would orphan them the same
# way — there is simply no namespace copy to route them to.
_BUILTIN_COPY: dict[type, Callable[[Any], Any]] = {
    label_types.Label: label.copy,
    line_types.Line: line.copy,
    box_types.Box: box.copy,
    matrix_types.Matrix: matrix.copy,
    list: array.copy,
    SequenceView: array.copy,
    dict: map_lib.copy,
}


def udt_copy(obj: Any, **changes: Any) -> Any:
    """
    Copy a Pine value the way its own type defines copying.

    Builtin objects are dispatched to their namespace ``copy()``; every other
    object is copied field by field with the given field overrides applied.
    Copying ``na`` yields ``na``.

    :param obj: The object to copy
    :param changes: Field values to override on the copy; user-defined types only
    :return: An independent copy of ``obj``, or ``obj`` itself when it is na
    :raises TypeError: If field overrides are passed for a builtin object
    """
    if isinstance(obj, NA):
        # Measured on TradingView: copying an na drawing leaves the registry
        # untouched and returns na. NA is interned per declared type, so this
        # value already IS the na the namespace copy would return.
        return obj
    builtin = _BUILTIN_COPY.get(type(obj))
    if builtin is None:
        return replace(obj, **changes)
    if changes:
        # Not expressible in Pine: the builtin ``.copy()`` takes no arguments, and
        # only ``@udt``'s own ``copy`` forwards changes, which lands on the branch
        # above. Refuse rather than poke attributes on a registered chart object.
        raise TypeError(f"{type(obj).__name__} copy takes no field overrides")
    return builtin(obj)


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
