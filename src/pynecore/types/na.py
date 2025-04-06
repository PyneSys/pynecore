from __future__ import annotations
from typing import Any, TypeVar, Generic, Type

T = TypeVar('T')


class NA(Generic[T]):
    """
    Class representing NA (Not Available) values.
    """
    __slots__ = ('type',)

    _type_cache: dict[Type, NA] = {}

    # noinspection PyShadowingBuiltins
    def __new__(cls, type: Type[T] | T | None = int) -> NA[T]:
        if type is None:
            return super().__new__(cls)
        try:
            # Use the cached instance if it exists
            return cls._type_cache[type]
        except KeyError:
            # Create a new instance and store it in the cache
            na = super().__new__(cls)
            cls._type_cache[type] = na
            return na

    # noinspection PyShadowingBuiltins
    def __init__(self, type: Type[T] | T | None = int):
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
        return f"NA[{self.type.__name__}]"

    def __str__(self) -> str:
        """
        Return a string representation of the NA value.
        """
        return ""

    def __hash__(self) -> int:
        """
        Return a hash value for the NA value.
        """
        return hash(self.type)

    def __int__(self) -> NA[int]:
        return NA(int)

    def __float__(self) -> NA[float]:
        return NA(float)

    def __bool__(self) -> bool:
        return False

    #
    # Arithmetic operations
    #

    def __neg__(self) -> NA[T]:
        return NA(self.type)

    def __add__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __radd__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __sub__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __rsub__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __mul__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __rmul__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __truediv__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __rtruediv__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __mod__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __rmod__(self, _: Any) -> NA[T]:
        return NA(self.type)

    def __abs__(self) -> NA[T]:
        return NA(self.type)

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

    def __getattr__(self, _: str) -> NA[T]:
        return self

    def __getitem__(self, _: Any) -> NA[T]:
        return self

    def __call__(self, *_, **__) -> NA[T]:
        return self
