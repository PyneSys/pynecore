from math import isfinite
from typing import Any, TypeVar, Generic, Type, Self

__all__ = [
    'NA', 'na_float', 'na_int', 'na_bool', 'na_str', 'isna_num',
]

T = TypeVar('T')

# The one float na: Pine's float-typed na IS a native IEEE-754 nan.
# Interned so identity-based fast paths (dict key lookup, ``is`` checks)
# always see the same object when the na came from ``NA(float)``.
_NAN = float('nan')


def isna_num(x: Any) -> bool:
    """
    Representation-agnostic na test for numeric (float-typed) values.

    Accepts both representations of a Pine numeric na: the ``NA`` object and a
    native non-finite float. Uses Pine's ``na()`` predicate semantics
    (``not isfinite``), so ``inf`` and ``-inf`` count as na too — matching
    TradingView, where ``na(inf)`` is ``true``.

    Cold paths only: never call this in a per-bar hot loop. Hot loops use the
    inline idioms ``x != x`` (nan-only) or ``isinstance(x, NA) or x != x``.

    :param x: value to test
    :return: ``True`` if ``x`` is a numeric na, ``False`` otherwise
    """
    if isinstance(x, NA):
        return True
    try:
        return not isfinite(x)
    except TypeError:
        return False


class NA(Generic[T]):
    """
    Class representing NA (Not Available) values for non-float types.

    ``NA(float)`` does NOT construct an instance: it returns the interned
    native ``float('nan')`` — Pine's float na is a real IEEE-754 nan, so
    arithmetic and comparisons on it run at native float speed. This shim in
    ``__new__`` is a permanent compatibility contract: every already-compiled
    script calling ``NA(float)`` (and the ``na(float)`` constructor face)
    transparently produces the native nan.

    All other types (int, str, drawing objects, UDTs, ...) get interned NA
    instances: every operation returns self, every comparison is False.
    """
    __slots__ = ('type',)

    _type_cache: dict[Type | None, 'NA'] = {}

    # noinspection PyShadowingBuiltins
    def __new__(cls, type: Type[T] | T | None = int) -> Self:
        if type is float:
            return _NAN  # type: ignore[return-value]
        try:
            return cls._type_cache[type]  # type: ignore[reportReturnType]
        except KeyError:
            na = super().__new__(cls)
            cls._type_cache[type] = na
            return na

    # noinspection PyShadowingBuiltins
    def __init__(self, type: Type[T] | T | None = int) -> None:
        """
        Initialize a new NA value with an optional type parameter.
        The default type is int.
        """
        self.type = type

    def __repr__(self) -> str:
        if self.type is None:
            return "NA"
        return f"NA[{self.type.__name__}]"  # type: ignore

    def __str__(self) -> str:
        return ""

    def __format__(self, format_spec: str) -> str:
        return "NaN"

    def __hash__(self) -> int:
        return hash(self.type)

    def __int__(self) -> 'NA[int]':
        # We solve this with an AST Transformer
        raise TypeError("NA cannot be converted to int")

    def __float__(self) -> 'NA[float]':
        # We solve this with an AST Transformer
        raise TypeError("NA cannot be converted to float")

    def __bool__(self) -> bool:
        return False

    def __round__(self, n=None) -> Self:
        return self

    #
    # Arithmetic operations — every operation propagates self
    #

    def __neg__(self) -> Self:
        return self

    def __add__(self, _: Any) -> Self:
        return self

    def __radd__(self, _: Any) -> Self:
        return self

    def __sub__(self, _: Any) -> Self:
        return self

    def __rsub__(self, _: Any) -> Self:
        return self

    def __mul__(self, _: Any) -> Self:
        return self

    def __rmul__(self, _: Any) -> Self:
        return self

    def __truediv__(self, _: Any) -> Self:
        return self

    def __rtruediv__(self, _: Any) -> Self:
        return self

    def __mod__(self, _: Any) -> Self:
        return self

    def __rmod__(self, _: Any) -> Self:
        return self

    def __abs__(self) -> Self:
        return self

    #
    # Bitwise operations
    #

    def __and__(self, _: Any) -> Self:
        return self

    def __rand__(self, _: Any) -> Self:
        return self

    def __or__(self, _: Any) -> Self:
        return self

    def __ror__(self, _: Any) -> Self:
        return self

    def __xor__(self, _: Any) -> Self:
        return self

    def __rxor__(self, _: Any) -> Self:
        return self

    def __lshift__(self, _: Any) -> Self:
        return self

    def __rlshift__(self, _: Any) -> Self:
        return self

    def __rshift__(self, _: Any) -> Self:
        return self

    def __rrshift__(self, _: Any) -> Self:
        return self

    def __invert__(self) -> Self:
        return self

    #
    # Comparisons — all False (Pine semantics)
    #

    def __eq__(self, _: Any) -> bool:
        return False

    def __ne__(self, _: Any) -> bool:
        return False

    def __gt__(self, _: Any) -> bool:
        return False

    def __lt__(self, _: Any) -> bool:
        return False

    def __le__(self, _: Any) -> bool:
        return False

    def __ge__(self, _: Any) -> bool:
        return False

    #
    # In contexts
    #

    def __getattr__(self, name: str) -> Self:
        # Don't return self for special attributes
        if name.startswith('__'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return self

    def __getitem__(self, _: Any) -> Self:
        return self

    def __contains__(self, _: Any) -> bool:
        # ``x in na`` must short-circuit to False: ``__getitem__`` returns self
        # for every index and never raises IndexError, so the ``in`` operator's
        # sequence-protocol fallback would iterate forever.
        return False

    def __iter__(self) -> Any:
        # NA is not a sequence. Without an explicit __iter__, ``iter(na)``
        # (e.g. ``str.join(na)``, ``list(na)``, tuple unpacking) falls back to
        # the sequence protocol and — since ``__getitem__`` returns self for
        # every index — spins forever while building an unbounded list.
        # Failing loudly is the only safe behavior here.
        raise TypeError("na is not iterable")

    def __call__(self, *_, **__) -> Self:
        return self


na_float: float = _NAN
na_int = NA(int)
na_str = NA(str)
na_bool = NA(bool)
