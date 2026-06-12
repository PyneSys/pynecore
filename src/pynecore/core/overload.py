"""
Function overloading with per-implementation instance state (slot scheme).

The ``@overload`` decorator registers every implementation under the
function's qualified name and binds the name to a single dispatcher. The
dispatcher selects the implementation by argument types (Pine Script
compatible matching) and calls it through a per-anchor bound cache:

- Call sites reach the dispatcher on the UNIFORM route — the caller anchors
  it in its own state vector with ``__bind_any__``, whose ``_bind_target``
  finds the dispatcher's ``__pyne_bind__`` factory and stores a fresh
  anchored dispatcher in the anchor slot.
- One anchor holds one bound callable PER IMPLEMENTATION: a call site where
  different argument types win on different bars keeps a separate persistent
  instance for every implementation, while one implementation's state
  persists across the bars it wins on.
- State-carrying implementations are bound through
  ``instance_state._bind_target`` (their ``__pyne_layout__`` comes from the
  ``@__attach_layout__`` decorator the slot transform inserts below
  ``@overload``); stateless implementations are called raw.
- Calling the dispatcher directly (no anchor — module level, non-transformed
  code, function values passed to builtins) falls back to the dispatcher's
  own module-lifetime bound cache: one shared instance per implementation,
  the same semantics the legacy module-global scope gave such calls.

Implementation matching skips the hidden state parameter the slot transform
injects (``__state__`` or the scope-qualified ``__state·{scope}__`` form):
signatures and parameter types are computed from the VISIBLE parameters
only, and the state argument is prepended by the bound partial, never by
the caller.
"""
from typing import (TypeVar, Callable, get_type_hints, overload as typing_overload,
                    Any, Type, Union, get_args, get_origin, cast)
from functools import wraps, partial
from inspect import signature
from collections import defaultdict
from types import FunctionType, UnionType

from .instance_state import _bind_target, _make_state, register_shared_cache
from ..types.base import StrLiteral
from ..types.na import NA

__all__ = ['overload']

T = TypeVar('T')


def _is_state_param(name: str) -> bool:
    """Whether a parameter is the hidden state parameter injected by the
    slot-layout transform.

    :param name: Parameter name.
    :return: True for ``__state__`` and the scope-qualified form.
    """
    return name == '__state__' or (name.startswith('__state·') and name.endswith('__'))


class Implementation:
    __slots__ = ('func', 'sig', 'type_hints', 'param_types')
    func: FunctionType
    sig: Any  # Signature object of the VISIBLE parameters
    type_hints: dict
    param_types: tuple  # Cached visible parameter types for quick checking

    def __init__(self, func: FunctionType):
        self.update(func)

    def update(self, func: FunctionType) -> None:
        """(Re)bind to the implementation function and cache its matching
        metadata. Re-running a module re-decorates the same source lines —
        the dispatcher and the Implementation objects survive, only the
        function objects are swapped.

        :param func: The (possibly re-created) implementation function.
        """
        if getattr(self, 'func', None) is not None and func.__code__ is self.func.__code__:
            # The same source line re-executed (library mains and nested
            # scopes re-run every bar): only the closure cells and default
            # values are new, the matching metadata is unchanged — skip the
            # expensive signature()/get_type_hints() recompute
            self.func = func
            return
        sig = signature(func)
        params = list(sig.parameters.values())
        if params and _is_state_param(params[0].name):
            # Hide the injected state parameter from matching: arity and
            # types are checked against what the call site passes, the
            # state argument comes from the bound partial
            params = params[1:]
        hints = get_type_hints(func)
        self.func = func
        self.sig = sig.replace(parameters=params)
        self.type_hints = hints
        self.param_types = tuple((p.name, hints.get(p.name, Any)) for p in params)


_registry: dict[str, list[Implementation]] = defaultdict(list)
_implementations: dict[str, Implementation] = {}  # Store implementations separately
_dispatchers: dict[str, Callable] = {}  # Store dispatchers separately


def _check_type(value: Any, expected_type: Type) -> bool:
    """Cached type checking for better performance with Pine Script compatibility"""
    # ``Any`` matches every value. Parameters without a type hint default to ``Any``
    # (see ``param_types`` below), and the compiler threads a closure variable in as a
    # leading, unannotated parameter -- both surface here as ``Any`` and must accept any
    # argument, like an unconstrained Pine parameter. isinstance() rejects ``Any``.
    if expected_type is Any:
        return True

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


def _select(impls: list[Implementation], args: tuple, kwargs: dict) -> Implementation | None:
    """Select the implementation matching a call's arguments.

    :param impls: Registered implementations (registration order).
    :param args: Positional arguments of the call.
    :param kwargs: Keyword arguments of the call.
    :return: The first matching implementation, or None.
    """
    # Quick path: try direct positional args match first
    if not kwargs:
        for impl in impls:
            if len(args) == len(impl.param_types):
                if all(_check_type(arg, type_)
                       for arg, (_, type_) in zip(args, impl.param_types)):
                    return impl

    # Slower path: handle mixed args/kwargs and defaults
    for impl in impls:
        try:
            bound = impl.sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if all(_check_type(value, impl.type_hints[name])
                   for name, value in bound.arguments.items()
                   if name in impl.type_hints):
                return impl
        except TypeError:
            continue
    return None


def _anchored(impls: list[Implementation], qualname: str,
              cache: dict[Implementation, tuple[Callable, list | None, Callable]] | None = None
              ) -> Callable:
    """Create an anchored dispatch entry with its own per-implementation
    bound cache. ``__pyne_bind__`` hands these out, one per anchor; the
    dispatcher itself is one too (the shared, anchorless fallback, whose
    cache is registered for clearing on ``instance_state.reset()`` — anchor
    caches die with their anchor, the dispatcher's would outlive the run).

    :param impls: The registry list of the overload group (shared, live).
    :param qualname: Qualified name for error messages.
    :param cache: Externally held cache dict (the dispatcher's registered
        one); per-anchor entries create their own.
    :return: The dispatch callable.
    """
    _cache: dict[Implementation, tuple[Callable, list | None, Callable]] = \
        {} if cache is None else cache

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        impl = _select(impls, args, kwargs)
        if impl is None:
            raise TypeError(f"No matching implementation found for {qualname}: {args}, {kwargs}")
        entry = _cache.get(impl)
        if entry is None or entry[0] is not impl.func:
            # First win at this anchor, or the implementation function was
            # re-created by a re-execution of its defining scope (library
            # mains re-run every bar). Keep the existing instance state and
            # take the closure from the new function object — the same
            # split pine_method._bound_method does
            func = impl.func
            layout: dict[str, Any] | None = getattr(func, '__pyne_layout__', None)
            if layout is not None:
                state = entry[1] if entry is not None and entry[1] is not None \
                    else _make_state(layout)
                entry = _cache[impl] = (func, state, partial(func, state))
            else:
                entry = _cache[impl] = (func, None, _bind_target(func))
        return entry[2](*args, **kwargs)

    return dispatch


def overload(func: Callable[..., T]) -> Callable[..., T]:
    """
    Function overloading decorator with:
    - Type checking cache
    - Pre-calculated signatures and type hints (hidden state parameter excluded)
    - Quick parameter matching
    - Per-anchor instance state through ``__pyne_bind__``
    - IDE type checking support via typing.overload
    """
    _func = cast(FunctionType, func)
    qualname = _func.__module__ + '.' + _func.__qualname__
    qualname_with_line = f"{qualname}:{_func.__code__.co_firstlineno}"

    # Re-executed module: same dispatcher, rebind the implementation
    _dispatcher = _dispatchers.get(qualname)
    if _dispatcher is not None:
        impl = _implementations.get(qualname_with_line)
        if impl is not None:
            impl.update(_func)
            return _dispatcher

    # Register with typing.overload for IDE support
    typing_overload(func)

    impl = Implementation(_func)
    _implementations[qualname_with_line] = impl
    _registry[qualname].append(impl)

    if _dispatcher is None:
        # The dispatcher must carry the implementation's metadata (__name__ in
        # particular): for exported library functions the @export decorator sits
        # above @overload and looks up the module-level Exported proxy by the
        # wrapped callable's __name__.
        _dispatcher = wraps(func)(_anchored(_registry[qualname], qualname,
                                            register_shared_cache({})))
        # @wraps copies the implementation's __dict__ too — including the
        # __pyne_layout__ the slot transform attached. The dispatcher must
        # not look state-carrying to the call-site classifier or to
        # _bind_target.
        _dispatcher.__dict__.pop('__pyne_layout__', None)
        setattr(_dispatcher, '__pyne_bind__',
                lambda: _anchored(_registry[qualname], qualname))
        _dispatchers[qualname] = _dispatcher

    return _dispatcher
