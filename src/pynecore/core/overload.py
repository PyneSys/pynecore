from typing import (TypeVar, Callable, get_type_hints, overload as typing_overload,
                    Any, Type, Union, get_args, get_origin, cast)
from inspect import signature
from collections import defaultdict
from types import FunctionType

from .function_isolation import isolate_function
from ..types.na import NA

__all__ = ['overload']

T = TypeVar('T')

__scope_id__ = ""


class Implementation:
    __slots__ = ('func', 'sig', 'type_hints', 'param_types')
    func: Callable
    sig: Any  # Signature object
    type_hints: dict
    param_types: tuple  # Cached parameter types for quick checking

    def __init__(self, func: Callable, sig: Any, type_hints: dict, param_types: tuple):
        self.func = func
        self.sig = sig
        self.type_hints = type_hints
        self.param_types = param_types


_registry: dict[str, list[Implementation]] = defaultdict(list)
_implementations: dict[str, Implementation] = {}  # Store implementations separately
_dispatchers: dict[str, Callable] = {}  # Store dispatchers separately


def _check_type(value: Any, expected_type: Type) -> bool:
    """Cached type checking for better performance with Pine Script compatibility"""
    # Direct type match
    if isinstance(value, expected_type):
        return True

    # Pine Script-like int to float conversion
    if expected_type is float and isinstance(value, int):
        return True

    # Handle NA values - Pine Script allows NA for any basic type
    if isinstance(value, NA):
        # Check if expected_type is a Pine Script basic type
        if expected_type in (int, float, str, bool):
            return True

        # For Union types containing basic types, NA is also acceptable
        origin = get_origin(expected_type)
        if origin in (Union, type(None) | type):
            args = get_args(expected_type)
            # If any of the Union members is a basic type, accept NA
            if any(arg in (int, float, str, bool) for arg in args):
                return True

        # For non-basic types, check if NA's type matches
        na_type = value.type
        # Handle the case when na_type is an actual instance and not a type
        if not isinstance(na_type, type):
            if na_type is None:
                return isinstance(None, expected_type)
            na_type = type(na_type)
        return na_type is expected_type

    # Handle Union types
    origin = get_origin(expected_type)
    if origin in (Union, type(None) | type):
        return any(_check_type(value, t) for t in get_args(expected_type))

    if hasattr(expected_type, '__instancecheck__'):
        return expected_type.__instancecheck__(value)

    return False


def overload(func: Callable[..., T]) -> Callable[..., T]:
    """
    Optimized function overloading decorator with:
    - Type checking cache
    - Pre-calculated signatures and type hints
    - Quick parameter matching
    - IDE type checking support via typing.overload
    """
    global __scope_id__

    _func = cast(FunctionType, func)
    qualname = _func.__module__ + '.' + _func.__qualname__
    qualname_with_line = f"{qualname}:{_func.__code__.co_firstlineno}"

    # This caching prevents re-creating the dispatcher if it already exists
    _dispatcher = _dispatchers.get(qualname)
    if _dispatcher:
        try:
            impl = _implementations[qualname_with_line]
            if impl:
                # Change the function implementation to the new one
                impl.func = func
                return _dispatcher
        except KeyError:
            pass

    # Register with typing.overload for IDE support
    typing_overload(func)

    # Pre-calculate and cache implementation info
    impl = Implementation(
        func=func,
        sig=signature(func),
        type_hints=get_type_hints(func),
        param_types=tuple(
            (name, get_type_hints(func).get(name, Any))
            for name in signature(func).parameters
        ),
    )
    _implementations[qualname_with_line] = impl

    if qualname not in _dispatchers:
        # noinspection PyShadowingNames
        def dispatcher(*args: Any, **kwargs: Any) -> Any:
            # Quick path: try direct positional args match first
            if not kwargs:
                for impl in _registry[qualname]:
                    if len(args) == len(impl.param_types):
                        if all(_check_type(arg, type_)
                               for arg, (_, type_) in zip(args, impl.param_types)):
                            return isolate_function(impl.func, '__overloaded__', __scope_id__)(*args)

            # Slower path: handle mixed args/kwargs
            for impl in _registry[qualname]:
                try:
                    bound = impl.sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    if all(_check_type(value, impl.type_hints.get(name, Any))
                           for name, value in bound.arguments.items()):
                        return isolate_function(impl.func, '__overloaded__', __scope_id__)(*args, **kwargs)
                except TypeError:
                    continue

            raise TypeError(f"No matching implementation found for {qualname}: {args}, {kwargs}")

        # Store implementation and dispatcher
        _registry[qualname].append(impl)

        _dispatcher = dispatcher

        _dispatchers[qualname] = _dispatcher
        return _dispatcher

    # Add additional implementation
    _registry[qualname].append(impl)

    dispatcher = _dispatchers[qualname]

    # Return existing dispatcher
    return dispatcher
