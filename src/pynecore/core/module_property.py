from typing import TypeVar, overload, Protocol, Any

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class ModulePropertyProtocol(Protocol[T_co]):
    @overload
    def __call__(self) -> T_co: ...

    @overload
    def __call__(self, *args: Any, **kwargs: Any) -> T_co: ...


def module_property(func) -> ModulePropertyProtocol[T] | T:
    """
    Decorator for Pine-style hybrid property/functions.
    """
    setattr(func, '__module_property__', True)
    return func
