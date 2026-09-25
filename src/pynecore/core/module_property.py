from typing import Any, Callable


def module_property(func: Callable[..., Any]) -> Any:
    """
    Decorator for Pine-style hybrid property/functions.

    Statically typed as the wrapped function's return value (see the stub): the AST
    transformer routes bare reads through the call, so user code sees a value.
    Use :func:`module_function_property` for the hybrids that user code also
    calls with arguments (``time(...)``, ``year(t)``, ``ta.tr(true)``, ...).
    """
    setattr(func, '__module_property__', True)
    return func


def module_function_property(func: Callable[..., Any]) -> Any:
    """
    Same runtime behavior as :func:`module_property`, for the Pine
    function-and-variable hybrids that are also called with arguments.

    Statically the result is both (see the stub): a value of the function's return
    type, and a callable with the function's own signature.
    """
    setattr(func, '__module_property__', True)
    return func
