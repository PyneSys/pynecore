from __future__ import annotations
from typing import Any, TypeVar, Generic, Type, Self

__all__ = ['NA', 'na_float', 'na_int', 'na_bool', 'na_str']

T = TypeVar('T')


class NA(Generic[T]):
    """
    Class representing NA (Not Available) values.
    """
    __slots__ = ('type',)

    _type_cache: dict[Type, NA] = {}

    # noinspection PyShadowingBuiltins
    def __new__(cls, type: Type[T] | T | None = int) -> Self:
        if type is None:
            return super().__new__(cls)
        try:
            # Use the cached instance if it exists
            return cls._type_cache[type]  # type: ignore[reportReturnType]
        except KeyError:
            # Create a new instance and store it in the cache
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
        """
        Return a string representation of the NA value.
        """
        if self.type is None:
            return "NA"
        return f"NA[{self.type.__name__}]"  # type: ignore

    def __str__(self) -> str:
        """
        Return a string representation of the NA value.
        """
        return ""

    def __format__(self, format_spec: str) -> str:
        return "NaN"

    def __hash__(self) -> int:
        """
        Return a hash value for the NA value.
        """
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
    # All comparisons should be false
    #

    def __eq__(self, _: Any) -> bool:
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

    def __call__(self, *_, **__) -> Self:
        return self


na_float = NA(float)
na_int = NA(int)
na_str = NA(str)
na_bool = NA(bool)
