from typing import Any, TypeVar, Callable

T = TypeVar('T')


def module_property(func: Callable[..., T]) -> T:
    """
    Decorator for Pine-style hybrid property/functions.

    Statically typed as the wrapped function's return value: the AST
    transformer routes bare reads through the call, so user code sees a value.
    Use :func:`module_function_property` for the hybrids that user code also
    calls with arguments (``time(...)``, ``year(t)``, ``ta.tr(true)``, ...) —
    the value/callable duality is not expressible, so those get ``Any``.
    """
    setattr(func, '__module_property__', True)
    return func  # type: ignore[return-value]


def module_function_property(func: Callable[..., T]) -> Any:
    """
    Same runtime behavior as :func:`module_property`, for the Pine
    function-and-variable hybrids that are also called with arguments.
    """
    setattr(func, '__module_property__', True)
    return func
