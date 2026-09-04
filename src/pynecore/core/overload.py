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

A call site whose argument TYPES the Pine type pass could decide reaches the
binder with an overload pin -- one type character per positional argument. The
anchor then turns those characters into witness values and runs the ORDINARY
selector on them once, at bind time, instead of selecting per bar from the
values. That is what makes the dispatch match TradingView's: Pine resolves an
overload from the static type, and the two answers differ exactly where an
int-TYPED expression carries a fractional value (``14 / 8`` is int-typed and
1.75 there), which the value-driven selector would widen to the float
implementation. Set ``PYNE_NO_TYPE_PIN=1`` to ignore every pin and dispatch
from the values alone.

Implementation matching skips the hidden state parameter the slot transform
injects (``__state__`` or the scope-qualified ``__state·{scope}__`` form):
signatures and parameter types are computed from the VISIBLE parameters
only, and the state argument is prepended by the bound partial, never by
the caller.
"""
import os
from typing import (TypeVar, Callable, get_type_hints, overload as typing_overload,
                    Any, Type, Union, get_args, get_origin, cast)
from functools import wraps, partial
from inspect import signature
from collections import defaultdict
from types import FunctionType, UnionType

from .instance_state import _bind_target, _make_state, register_shared_cache, __dyn_default__  # noqa: internal API
from ..types.base import StrLiteral
from ..types.matrix import Matrix
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


def _match_declared(declared: Any, expected: Any, strict: bool) -> bool:
    """Whether a declared type satisfies an expected one, structurally.

    :param declared: The type an na argument carries, or one of its arguments.
    :param expected: The declared parameter type, or one of its arguments.
    :param strict: Match without the int-where-float-is-wanted widening.
    """
    # Type arguments cannot be compared by identity: two independently built
    # generic aliases are equal but not the same object (`list[int] is
    # list[int]` is False -- typing caches its own aliases, so a user Generic
    # would accidentally pass while a builtin one never does), and a nested
    # subscript has to be matched argument by argument
    if expected is Any or declared is expected:
        return True
    if not strict and expected is float and declared is int:
        return True
    if (get_origin(declared) or declared) is not (get_origin(expected) or expected):
        return False
    expected_args = get_args(expected)
    declared_args = get_args(declared)
    # A side without a subscript carries no element type at all, so it takes any
    # parameterization -- the same permissiveness an empty container gets when
    # it is sampled. Two subscripts of DIFFERENT arity are genuinely different
    # shapes though, and must not match
    if not expected_args or not declared_args:
        return True
    if len(declared_args) != len(expected_args):
        return False
    return all(_match_declared(a, b, strict) for a, b in zip(declared_args, expected_args))


def _check_type(value: Any, expected_type: Type, strict: bool = False) -> bool:
    """Cached type checking for better performance with Pine Script compatibility

    :param value: A call argument.
    :param expected_type: The parameter's declared type.
    :param strict: Match without the int-where-float-is-wanted widening, so an
        int argument takes an int parameter over a float one. ``_select`` runs
        this pass first.
    """
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
    _origin: Any = get_origin(expected_type)
    # The element types of a parameterized expectation, kept for the na branch
    # below: matching an na argument needs them AFTER expected_type has been
    # stripped to its origin here
    _expected_args: tuple = ()
    if isinstance(_origin, type) and _origin is not UnionType:
        if isinstance(value, _origin):
            _args = get_args(expected_type)
            if _args and isinstance(value, dict):
                if value:
                    _key, _val = next(iter(value.items()))
                    return (_check_type(_key, _args[0], strict)
                            and _check_type(_val, _args[1], strict))
            elif _args and isinstance(value, (list, tuple)) and value:
                return _check_type(value[0], _args[0], strict)
            elif _args and isinstance(value, Matrix) and value.data and value.data[0]:
                # A matrix keeps no element type of its own, so it is sampled
                # like a list is
                return _check_type(value.data[0][0], _args[0], strict)
            return True
        _expected_args = get_args(expected_type)
        expected_type = cast(Type, _origin)

    # Unions (`int | float` and typing.Union alike): isinstance() rejects a union
    # holding a parameterized generic, and an na argument has to be matched
    # against the members one by one anyway
    elif _origin is UnionType or _origin is Union:
        return any(_check_type(value, t, strict) for t in get_args(expected_type))

    # Direct type match
    if isinstance(value, expected_type):
        # Python's bool subclasses int, Pine's does not: a bool argument must
        # not answer an int parameter
        if expected_type is int and (value is True or value is False):
            return False
        return True

    # Pine Script-like int to float conversion. ``type(value) is int`` and not
    # isinstance(): a bool is an int in Python, and Pine never widens it either
    if not strict and expected_type is float and type(value) is int:
        return True

    # The mirror image, and it is not symmetry but a measured TradingView law:
    # ``int / int`` is int-TYPED while keeping its fractional VALUE. Measured on
    # BINANCE:BTCUSDT 30m in v4 and v6 alike, with ``R = 14``: ``R / 8`` plots
    # 1.75 and ``R / 8 * 100`` plots 175, yet ``ta.highest(R / 8)`` compiles and
    # equals ``ta.highest(1)`` and ``ta.sma(close, R / 8)`` equals
    # ``ta.sma(close, 1)`` — the truncation happens where an integer is actually
    # required, not at the division. Such a value reaches an int parameter as a
    # plain Python float, so the exact pass above cannot place it; the callee
    # truncates it (``length = int(length)``), exactly as TradingView does.
    if not strict and expected_type is int and type(value) is float:
        return True

    # Pine Script allows plain str where StrLiteral subtypes are expected (e.g. size, xloc)
    if isinstance(value, str) and isinstance(expected_type, type) and issubclass(expected_type, StrLiteral):
        return True

    # Handle NA values - Pine Script allows NA for any basic type
    if isinstance(value, NA):
        na_type = value.type
        # A typeless `na` is assignable to anything, like in Pine
        if na_type is None:
            return True

        # Check if expected_type is a Pine Script basic type
        if expected_type in (int, float, str, bool):
            return not strict or na_type is expected_type

        # An na of a container carries its declared type whole, subscript
        # included (`matrix<float> m = na` gives NA(Matrix[float])) -- that
        # subscript is the only element type such an argument has, so it is
        # what discriminates two overloads of the same container here
        na_args = get_args(na_type)
        if na_args:
            if get_origin(na_type) is not expected_type:
                return False
            if not _expected_args:
                return True
            if len(na_args) != len(_expected_args):
                return False
            return all(_match_declared(a, b, strict)
                       for a, b in zip(na_args, _expected_args))
        # Handle the case when na_type is an actual instance and not a type
        if not isinstance(na_type, type):
            na_type = type(na_type)
        return na_type is expected_type

    if hasattr(expected_type, '__instancecheck__'):
        return expected_type.__instancecheck__(value)

    return False


def _select(impls: list[Implementation], args: tuple, kwargs: dict) -> Implementation | None:
    """Select the implementation matching a call's arguments.

    An exact pass runs before the widening one, so the declaration order only
    decides between implementations that match equally well.

    :param impls: Registered implementations (registration order).
    :param args: Positional arguments of the call.
    :param kwargs: Keyword arguments of the call.
    :return: The first matching implementation, or None.
    """
    # Measured on TradingView (FX:EURUSD 240): with `f(float)` and `f(int)`
    # declared in either order, an int argument -- literal, variable, series or
    # a typed na -- takes the int one, and only a script without an int
    # implementation at all widens it to the float one. The same holds for the
    # element type of an array or matrix argument.
    for strict in (True, False):
        # Quick path: try direct positional args match first
        if not kwargs:
            for impl in impls:
                if len(args) == len(impl.param_types):
                    if all(_check_type(arg, type_, strict)
                           for arg, (_, type_) in zip(args, impl.param_types)):
                        return impl

        # Slower path: handle mixed args/kwargs and defaults
        for impl in impls:
            try:
                bound = impl.sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # ``__dyn_default__`` marks a parameter DynamicDefaultTransformer
                # took over: the declared default referenced ``lib.*`` (``= na``
                # above all), so the real value is computed in the body when the
                # argument is omitted. The sentinel is a bare object() and matches
                # no annotation -- type-checking it would reject every overload
                # whose optional parameters the caller left out.
                if all(_check_type(value, impl.type_hints[name], strict)
                       for name, value in bound.arguments.items()
                       if name in impl.type_hints and value is not __dyn_default__):
                    return impl
            except TypeError:
                continue
    return None


#: Value each pin character stands for. A pin records the STATIC types the
#: type pass decided on; the witnesses turn them back into values the ordinary
#: :func:`_select` understands, so the static and the dynamic decision run the
#: SAME code and cannot drift apart. Only types with one unambiguous witness
#: are pinnable, which is why there is no entry for a container or a drawing.
_PIN_WITNESSES: dict[str, Any] = {'i': 0, 'f': 0.0, 'b': True, 's': ''}

#: Pin character for a position with no witness — a container, a drawing, a
#: user type, or a type the pass could not settle. The position carries no
#: information, so it is left out of the selection instead of blocking it:
#: an int argument next to a user-typed one is exactly the shape the pin
#: exists for.
_PIN_ANY = '*'


def _select_pinned(impls: list[Implementation], pin: str) -> Implementation | None:
    """Select the implementation a pin with wildcard positions names.

    The witnessed positions are matched the way :func:`_select` matches them,
    the wildcard ones are not looked at. That is only an answer while it is the
    ONLY answer: where more than one implementation survives, the position the
    pin knows nothing about is the one that decides, and the values know it and
    the pin does not.

    :param impls: Registered implementations (registration order).
    :param pin: The call site's pin, one character per positional argument.
    :return: The single matching implementation, or None.
    """
    argc = len(pin)
    probes = [(index, _PIN_WITNESSES[char])
              for index, char in enumerate(pin) if char != _PIN_ANY]
    for strict in (True, False):
        matches = [impl for impl in impls
                   if len(impl.param_types) == argc
                   and all(_check_type(value, impl.param_types[index][1], strict)
                           for index, value in probes)]
        if matches:
            return matches[0] if len(matches) == 1 else None
    return None


def _type_token(value: Any) -> Any:
    """Hashable token capturing exactly the properties ``_check_type``
    discriminates on, so that two arguments with equal tokens are
    interchangeable for implementation selection.

    Scalars map to their type; NA carries its ``type`` marker; containers
    tokenize one sampled element, mirroring ``_check_type``'s element probe --
    recursively, because the sample can be an na or a container itself, and
    ``type()`` alone would merge ``NA(int)`` with ``NA(str)``.
    The token is conservative: distinct tokens never merge arguments that
    ``_check_type`` could treat differently.

    :param value: A call argument.
    :return: A hashable selection token.
    """
    t = type(value)
    if t is int or t is float or t is str or t is bool:
        return t
    if isinstance(value, NA):
        return NA, value.type
    if t is dict:
        if value:
            _k, _v = next(iter(value.items()))
            return dict, _type_token(_k), _type_token(_v)
        return (dict,)
    if t is list or t is tuple:
        if value:
            return t, _type_token(value[0])
        return (t,)
    if t is Matrix:
        if value.data and value.data[0]:
            return Matrix, _type_token(value.data[0][0])
        return (Matrix,)
    return t


def _canonical_kwarg_renames(impls: list[Implementation],
                             names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Compute the keyword renames onto canonically renamed parameters.

    An untyped call site emits a keyword argument under its original Pine
    spelling while the library ``def`` declares ``name + '__ren__'``
    (PyneComp's canonical rename; the same contract as
    ``pine_method._adapt_exported_kwargs``, which cannot see through the
    dispatcher). A keyword is renamed only when NO implementation declares
    the raw name and at least one declares the suffixed image — a correct
    call is never altered. Overloads of one export come from one compilation
    unit, so a name's rename decision is identical across implementations.

    :param impls: Registered implementations of the overload group.
    :param names: Keyword argument names as emitted at the call site.
    :return: ``(raw, canonical)`` pairs to apply; empty when nothing renames.
    """
    declared = {name for impl in impls for name, _ in impl.param_types}
    return tuple((k, k + '__ren__') for k in names
                 if k not in declared and k + '__ren__' in declared)


def _anchored(impls: list[Implementation], qualname: str,
              cache: dict[Implementation, tuple[Callable, list | None, Callable]] | None = None,
              pin: str | None = None) -> Callable:
    """Create an anchored dispatch entry with its own per-implementation
    bound cache. ``__pyne_bind__`` hands these out, one per anchor; the
    dispatcher itself is one too (the shared, anchorless fallback, whose
    cache is registered for clearing on ``instance_state.reset()`` — anchor
    caches die with their anchor, the dispatcher's would outlive the run).

    :param impls: The registry list of the overload group (shared, live).
    :param qualname: Qualified name for error messages.
    :param cache: Externally held cache dict (the dispatcher's registered
        one); per-anchor entries create their own.
    :param pin: The call site's overload pin, when the type pass decided on
        one: the implementation is then selected here, once, from the static
        types rather than per bar from the values.
    :return: The dispatch callable.
    """
    _cache: dict[Implementation, tuple[Callable, list | None, Callable]] = \
        {} if cache is None else cache
    # Per-anchor selection memo: a call site invokes the dispatcher with the
    # same argument shape every bar, so the matching implementation is cached
    # by call shape and the full _select (with inspect binding) runs once.
    # Implementation objects are module-lifetime stable (re-runs swap only
    # impl.func, handled below), so this never goes stale across resets.
    _select_cache: dict[tuple, Implementation] = {}
    # Keyword-name adaptation memo (see _canonical_kwarg_renames): keyed by
    # the call's keyword-name tuple, stable for the same reason _select_cache
    # is — implementation signatures only change with a recompile.
    _kw_rename_cache: dict[tuple[str, ...], tuple[tuple[str, str], ...]] = {}
    # Positional-only shortcut over _select_cache, keyed by the raw argument
    # TYPES: a hit needs one map(type) and one dict lookup, with no _type_token
    # call and no nested key tuple to build. Latched off for good on the first
    # call whose tokens are not simply its types (see below), so an anchor that
    # passes na or container arguments does not keep paying for a key it can
    # never hit on. A one-element list, not a nonlocal: the read is the hot
    # part and a list index is cheaper than a cell rebind would be worth.
    _fast: dict[tuple[type, ...], Implementation] = {}
    _fast_ok = [True]

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        impl = types = None
        if _fast_ok[0] and not kwargs:
            types = tuple(map(type, args))
            impl = _fast.get(types)
        if impl is None:
            if kwargs:
                names = tuple(kwargs)
                renames = _kw_rename_cache.get(names)
                if renames is None:
                    renames = _kw_rename_cache[names] = _canonical_kwarg_renames(impls, names)
                for raw, canonical in renames:
                    kwargs[canonical] = kwargs.pop(raw)
            # Selection key, inlined (this is per-bar hot code): a uniform hashable
            # ``(positional_tokens, keyword_tokens)`` pair so the no-kwargs and
            # with-kwargs forms can never collide. map() over _type_token beats a
            # generator expression here. Equal keys guarantee the same impl.
            pos = tuple(map(_type_token, args))
            key = (pos, ()) if not kwargs else \
                (pos, tuple((k, _type_token(v)) for k, v in kwargs.items()))
            impl = _select_cache.get(key)
            if impl is None:
                impl = _select(impls, args, kwargs)
                if impl is None:
                    raise TypeError(f"No matching implementation found for {qualname}: {args}, {kwargs}")
                _select_cache[key] = impl
            if types is not None:
                # A token that IS its argument's type carries nothing the type does
                # not: _type_token picks its branch on type() alone, and the branches
                # returning the bare type are exactly the ones that look no further,
                # so every value of such a type tokenizes identically and the type
                # tuple selects what the token key would. Where an argument tokenized
                # to a tuple instead — an na with its declared type, a container with
                # its sampled element — the type tuple would merge arguments
                # _check_type can tell apart, so it must never key this anchor.
                if pos == types:
                    _fast[types] = impl
                else:
                    _fast_ok[0] = False
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

    # The bound cache is otherwise opaque to state walkers; the loop-site
    # same-bar rollback (``instance_state._collect_bound_builtins``) reads the
    # per-implementation state vectors through this reference. The registry
    # list rides along so a module can reach its implementations' layouts
    # after the dispatcher has replaced the def name (the ``per_call`` marks
    # in ``lib.ta`` need exactly that).
    dispatch.__pyne_cache__ = _cache
    dispatch.__pyne_impls__ = impls

    if not pin or os.environ.get('PYNE_NO_TYPE_PIN') == '1':
        return dispatch

    if _PIN_ANY in pin:
        pinned = _select_pinned(impls, pin) \
            if all(c in _PIN_WITNESSES or c == _PIN_ANY for c in pin) else None
    else:
        witnesses = tuple(_PIN_WITNESSES[c] for c in pin if c in _PIN_WITNESSES)
        pinned = _select(impls, witnesses, {}) if len(witnesses) == len(pin) else None
    if pinned is None:
        # An unwitnessable character, or no implementation for the static
        # shape: the values know more than the pin does, so let them decide,
        # exactly as they did before there were pins
        return dispatch
    # Narrowed alias: the closure below reads it on every call, and only the
    # non-None case ever gets there
    chosen: Implementation = pinned
    argc = len(pin)
    group = len(impls)
    # Positions the pin carries no type for. The selection ignored them, so the
    # first call checks the values against what the chosen implementation
    # declares there: a witnessed position can name an implementation the
    # arguments as a whole do not fit, and the values must win that.
    wildcards = tuple(index for index, char in enumerate(pin) if char == _PIN_ANY)
    verified = [not wildcards]

    def dispatch_pinned(*args: Any, **kwargs: Any) -> Any:
        if kwargs or len(args) != argc or len(impls) != group:
            # Not the shape the pin was computed for — a keyword spelling, a
            # different arity, or an implementation registered after the bind
            return dispatch(*args, **kwargs)
        if not verified[0]:
            if not all(_check_type(args[index], chosen.param_types[index][1], False)
                       for index in wildcards):
                return dispatch(*args, **kwargs)
            verified[0] = True
        # The entry bookkeeping of ``dispatch``, repeated rather than shared:
        # this is per-bar code, and the point of the pinned route is that it
        # costs one dict lookup and nothing else — no type tuple, no token,
        # no selection
        entry = _cache.get(chosen)
        if entry is None or entry[0] is not chosen.func:
            func = chosen.func
            layout: dict[str, Any] | None = getattr(func, '__pyne_layout__', None)
            if layout is not None:
                state = entry[1] if entry is not None and entry[1] is not None \
                    else _make_state(layout)
                entry = _cache[chosen] = (func, state, partial(func, state))
            else:
                entry = _cache[chosen] = (func, None, _bind_target(func))
        return entry[2](*args)

    dispatch_pinned.__pyne_cache__ = _cache
    dispatch_pinned.__pyne_impls__ = impls
    return dispatch_pinned


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
                lambda pin=None: _anchored(_registry[qualname], qualname, pin=pin))
        _dispatchers[qualname] = _dispatcher

    return _dispatcher
