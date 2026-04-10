from __future__ import annotations
from typing import Any, TypeVar, Generic, Type, Self

__all__ = [
    'NA', 'na_float', 'na_int', 'na_bool', 'na_str',
    'na_inf', 'na_neg_inf', 'na_nan',
]

T = TypeVar('T')


# Sentinel "types" used as markers for Pine-native special float values.
# We never instantiate these; they only serve as unique identifiers stored
# in NA.type and cached by _type_cache.
class _InfType:
    """Sentinel marking an NA that arithmetically behaves as +inf."""
    _na_value = float('inf')
    __name__ = 'inf'


class _NegInfType:
    """Sentinel marking an NA that arithmetically behaves as -inf."""
    _na_value = float('-inf')
    __name__ = 'neg_inf'


class _NanType:
    """Sentinel marking an NA that arithmetically behaves as nan."""
    _na_value = float('nan')
    __name__ = 'nan'


_INF = float('inf')
_NEG_INF = float('-inf')
_NAN = float('nan')


def _resolve(x: Any) -> Any:
    """Unwrap a valued NA to its backing float; pass through everything else."""
    if isinstance(x, NA):
        return getattr(x.type, '_na_value', None)
    return x


def _wrap(result: float) -> Any:
    """Normalize a float result back to one of the NA singletons, or NA(float)."""
    if result != result:  # nan
        return na_nan
    if result == _INF:
        return na_inf
    if result == _NEG_INF:
        return na_neg_inf
    # Finite result from an inf/nan operand should not happen under IEEE-754
    return NA(float)


class NA(Generic[T]):
    """
    Class representing NA (Not Available) values.

    NA can optionally carry a Pine-native special value marker (inf, -inf, nan)
    via its `type` slot pointing to one of the sentinel types (_InfType, etc.).
    Valued NAs still report True under `isinstance(x, NA)` and Pine's `na()`
    predicate — preserving the lib-internal NA-skip semantics — while their
    comparison and arithmetic operators follow IEEE-754 so that user-level
    Pine expressions (e.g. `inf > 40`) match TradingView behavior.
    """
    __slots__ = ('type',)

    _type_cache: dict[Type, NA] = {}

    # noinspection PyShadowingBuiltins
    def __new__(cls, type: Type[T] | T | None = int) -> Self:
        if type is None:
            return super().__new__(cls)
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

    def __int__(self) -> NA[int]:
        # We solve this with an AST Transformer
        raise TypeError("NA cannot be converted to int")

    def __float__(self) -> NA[float]:
        # We solve this with an AST Transformer
        raise TypeError("NA cannot be converted to float")

    def __bool__(self) -> bool:
        return False

    def __round__(self, n=None):
        return self

    #
    # Arithmetic operations
    #

    def __neg__(self) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        return _wrap(-v)

    def __add__(self, other: Any) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        o = _resolve(other)
        if o is None:
            return self
        try:
            return _wrap(v + o)
        except TypeError:
            return self

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        o = _resolve(other)
        if o is None:
            return self
        try:
            return _wrap(v - o)
        except TypeError:
            return self

    def __rsub__(self, other: Any) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        o = _resolve(other)
        if o is None:
            return self
        try:
            return _wrap(o - v)
        except TypeError:
            return self

    def __mul__(self, other: Any) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        o = _resolve(other)
        if o is None:
            return self
        try:
            return _wrap(v * o)
        except TypeError:
            return self

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        o = _resolve(other)
        if o is None:
            return self
        try:
            return _wrap(v / o)
        except (TypeError, ZeroDivisionError):
            return self

    def __rtruediv__(self, other: Any) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        o = _resolve(other)
        if o is None:
            return self
        try:
            return _wrap(o / v)
        except (TypeError, ZeroDivisionError):
            return self

    def __mod__(self, _: Any) -> Self:
        return self

    def __rmod__(self, _: Any) -> Self:
        return self

    def __abs__(self) -> Any:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return self
        return _wrap(abs(v))

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
    # Comparisons
    #
    # Plain NA: all comparisons False (Pine semantics).
    # Valued NA: IEEE-754 semantics against the backing float.
    #

    def __eq__(self, other: Any) -> bool:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return False
        o = _resolve(other)
        if o is None:
            return False
        try:
            return v == o
        except TypeError:
            return False

    def __ne__(self, other: Any) -> bool:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return False
        o = _resolve(other)
        if o is None:
            return False
        try:
            return v != o
        except TypeError:
            return False

    def __gt__(self, other: Any) -> bool:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return False
        o = _resolve(other)
        if o is None:
            return False
        try:
            return v > o
        except TypeError:
            return False

    def __lt__(self, other: Any) -> bool:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return False
        o = _resolve(other)
        if o is None:
            return False
        try:
            return v < o
        except TypeError:
            return False

    def __le__(self, other: Any) -> bool:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return False
        o = _resolve(other)
        if o is None:
            return False
        try:
            return v <= o
        except TypeError:
            return False

    def __ge__(self, other: Any) -> bool:
        v: float | None = getattr(self.type, '_na_value', None)
        if v is None:
            return False
        o = _resolve(other)
        if o is None:
            return False
        try:
            return v >= o
        except TypeError:
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

    def __call__(self, *_, **__) -> Self:
        return self


na_float = NA(float)
na_int = NA(int)
na_str = NA(str)
na_bool = NA(bool)

# Pine-native special float singletons. Cached by _type_cache through __new__.
na_inf = NA(_InfType)
na_neg_inf = NA(_NegInfType)
na_nan = NA(_NanType)
