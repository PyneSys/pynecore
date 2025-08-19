from typing import Callable
from types import FunctionType
from dataclasses import is_dataclass, replace as dataclass_replace
from copy import copy
from .pine_export import Exported
from .series import SeriesImpl

__all__ = ['isolate_function', 'reset']

# Store all function instances
_function_cache: dict[str | tuple, FunctionType] = {}


def reset():
    """
    Reset all function instances
    """
    _function_cache.clear()


def isolate_function(
        func: FunctionType | Callable, call_id: str | None, parent_scope: str,
        closure_argument_count: int = -1, call_counter: int = 0
) -> FunctionType | Callable:
    """
    Create a new function instance with isolated globals if the function has persistent or series globals.

    :param func: The function to create an instance of
    :param call_id: The unique call ID
    :param parent_scope: The parent scope ID
    :param closure_argument_count: Whether the function has closure arguments
    :param call_counter: The current call counter value for this call_id
    :return: The new function instance if there are any persistent or series globals otherwise the original function
    """
    # If there is no call ID, return the function as is
    if call_id is None:
        return func  # type: ignore

    # If it is a type object, return it as is
    if isinstance(func, type):
        return func  # type: ignore

    # If it is a classmethod (bound method where __self__ is a class), return it as is
    if hasattr(func, '__self__') and isinstance(func.__self__, type):
        return func  # type: ignore

    # Check if this is an Exported proxy and unwrap it
    if isinstance(func, Exported):
        unwrapped_func = func.__fn__
        if unwrapped_func is None:
            raise ValueError("Exported proxy has not been initialized with a function yet")
        func = unwrapped_func

    # If it is an overloaded function, returned by the dispatcher
    is_overloaded = call_id == '__overloaded__?'

    # Create full call ID from parent scope and call ID
    if call_id and not is_overloaded:
        # call_id_key = f"{parent_scope}→{call_id}#{call_counter}"
        call_id_key: str | tuple = (parent_scope, call_id, call_counter)
    else:
        call_id_key = parent_scope

    # If the function is overloaded, we need to remove the dispatcher from the cache to override it with implementation
    if is_overloaded:
        del _function_cache[call_id_key]

    try:
        # If a function is cached we can just call it
        isolated_function = _function_cache[call_id_key]

        if closure_argument_count == -1:  # If closures have been converted to  arguments, no closure is needed
            # We need to create new instance in every run only if the function is inside the main function
            # Create a new function with original closure and isolated globals
            isolated_function = FunctionType(
                func.__code__,
                isolated_function.__globals__,
                func.__name__,
                func.__defaults__,
                func.__closure__
            )

        return isolated_function
    except KeyError:
        pass

    # Builtin objects have no __globals__ attribute
    try:
        new_globals = dict(func.__globals__)
    except AttributeError:  # This is a builtin function (it should be filtered in the transformer)
        return func  # type: ignore

    # The qualified name of the function, this name is used in the globals registry by transformer
    qualname = func.__qualname__.replace('<locals>.', '')

    # If globals are registered, we can use them
    registry_found = False
    try:
        persistent_vars = new_globals['__persistent_function_vars__']
        registry_found = True
    except KeyError:
        persistent_vars = {}
    try:
        series_vars = new_globals['__series_function_vars__']
        registry_found = True
    except KeyError:
        series_vars = {}

    try:
        for key in persistent_vars[qualname]:
            old_value = new_globals[key]
            if isinstance(old_value, (dict, list)):
                new_globals[key] = old_value.copy()
            elif is_dataclass(old_value):
                new_globals[key] = dataclass_replace(old_value)  # type: ignore
            else:
                new_globals[key] = copy(old_value)
    except KeyError:
        pass
    try:
        for key in series_vars[qualname]:
            old_value = new_globals[key]
            new_globals[key] = SeriesImpl(old_value._max_bars_back)  # noqa
    except KeyError:
        pass

    # Fallback, if globals are not registered
    if not registry_found:
        # Create new globals with isolated persistent and series
        for key in new_globals.keys():
            if key.startswith('__persistent_') and not key.endswith('_vars__'):
                old_value = new_globals[key]
                if isinstance(old_value, (dict, list)):
                    new_globals[key] = old_value.copy()
                elif is_dataclass(old_value):
                    new_globals[key] = dataclass_replace(old_value)  # type: ignore
                else:
                    new_globals[key] = copy(old_value)
            elif key.startswith('__series_') and not key.endswith('_vars__'):
                old_value = new_globals[key]
                new_globals[key] = type(old_value)(old_value._max_bars_back)  # noqa

    new_globals['__scope_id__'] = call_id_key

    # Create a new function with new closure and globals
    isolated_function = FunctionType(
        func.__code__,
        new_globals,
        func.__name__,
        func.__defaults__,
        func.__closure__
    )

    _function_cache[call_id_key] = isolated_function
    return isolated_function
