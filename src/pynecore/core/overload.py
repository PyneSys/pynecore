from typing import (TypeVar, Callable, get_type_hints, overload as typing_overload,
                    Any, Type, Union, get_args, get_origin, cast)
from functools import wraps
from inspect import signature
from collections import defaultdict
from types import FunctionType, UnionType

from .function_isolation import isolate_function
from ..types.base import StrLiteral
from ..types.na import NA

__all__ = ['overload']

T = TypeVar('T')

__scope_id__ = ""


class Implementation:
    __slots__ = ('func', 'sig', 'type_hints', 'param_types', 'call_id')
    func: Callable
    sig: Any  # Signature object
    type_hints: dict
    param_types: tuple  # Cached parameter types for quick checking
    call_id: str  # Per-implementation isolation call id

    def __init__(self, func: Callable, sig: Any, type_hints: dict, param_types: tuple,
                 call_id: str):
        self.func = func
        self.sig = sig
        self.type_hints = type_hints
        self.param_types = param_types
        self.call_id = call_id


_registry: dict[str, list[Implementation]] = defaultdict(list)
_implementations: dict[str, Implementation] = {}  # Store implementations separately
_dispatchers: dict[str, Callable] = {}  # Store dispatchers separately


def _check_type(value: Any, expected_type: Type) -> bool:
    """Cached type checking for better performance with Pine Script compatibility"""
    # Parameterized containers (list[T], dict[K, V], ...): isinstance() rejects
    # parameterized generics. Match on the container type, then discriminate on a
    # sample element -- overloads can differ only in their element types
    # (map<string, string> vs map<string, float>)
    _origin = get_origin(expected_type)
    if isinstance(_origin, type) and _origin is not UnionType:
        if isinstance(value, _origin):
            _args = get_args(expected_type)
            if _args and isinstance(value, dict):
                if value:
                    _key, _val = next(iter(value.items()))
                    return _check_type(_key, _args[0]) and _check_type(_val, _args[1])
            elif _args and isinstance(value, (list, tuple)) and value:
                return _check_type(value[0], _args[0])
            return True
        expected_type = cast(Type, _origin)

    # Direct type match
    if isinstance(value, expected_type):
        return True

    # Pine Script-like int to float conversion
    if expected_type is float and isinstance(value, int):
        return True

    # Pine Script allows plain str where StrLiteral subtypes are expected (e.g. size, xloc)
    if isinstance(value, str) and isinstance(expected_type, type) and issubclass(expected_type, StrLiteral):
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
        # A typeless `na` is assignable to anything, like in Pine
        if na_type is None:
            return True
        # Handle the case when na_type is an actual instance and not a type
        if not isinstance(na_type, type):
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

    # Pre-calculate and cache implementation info; the call id carries the
    # implementation's identity so different overloads never share one
    # isolation cache slot (a shared slot would rebuild one implementation's
    # code with another one's globals)
    impl = Implementation(
        func=func,
        sig=signature(func),
        type_hints=get_type_hints(func),
        param_types=tuple(
            (name, get_type_hints(func).get(name, Any))
            for name in signature(func).parameters
        ),
        call_id=f"__overloaded__·{qualname_with_line}",
    )
    _implementations[qualname_with_line] = impl

    if qualname not in _dispatchers:
        # The dispatcher must carry the implementation's metadata (__name__ in particular):
        # for exported library functions the @export decorator sits above @overload and looks
        # up the module-level Exported proxy by the wrapped callable's __name__.
        # noinspection PyShadowingNames
        @wraps(func)
        def dispatcher(*args: Any, **kwargs: Any) -> Any:
            # Quick path: try direct positional args match first
            if not kwargs:
                for impl in _registry[qualname]:
                    if len(args) == len(impl.param_types):
                        if all(_check_type(arg, type_)
                               for arg, (_, type_) in zip(args, impl.param_types)):
                            return isolate_function(impl.func, impl.call_id, __scope_id__)(*args)

            # Slower path: handle mixed args/kwargs
            for impl in _registry[qualname]:
                try:
                    bound = impl.sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    if all(_check_type(value, impl.type_hints[name])
                           for name, value in bound.arguments.items()
                           if name in impl.type_hints):
                        return isolate_function(impl.func, impl.call_id, __scope_id__)(*args, **kwargs)
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
