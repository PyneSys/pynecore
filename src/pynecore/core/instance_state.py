"""
Runtime core of the slot-based instance state scheme.

Function-instance state lives in plain lists ("state vectors") whose slots are
assigned at transform time; the emitted code addresses them with literal int
indexes. A child instance (the state of an isolated call site) occupies a
dedicated slot of its PARENT's state vector, so all live state forms a tree
hanging off a small set of root vectors (script main, library mains, security
processes). There is no global keyed instance cache: dropping a parent
releases its whole subtree through normal GC.

This module is the successor of the deleted ``function_isolation.py``
runtime (module-globals copying with a global keyed instance cache).

Layout metadata
---------------

The transformer emits one ``__pyne_slot_layout__`` dict per module (one entry
per scope) and attaches the matching entry to every state-carrying function
as ``func.__pyne_layout__``. An entry is a plain dict with these keys:

``init``
    Tuple with the template value of every slot; ``list(init)`` is the
    instantiation. The values are immutable by construction (literals or
    ``NA``; non-literal initializers go through the lazy init-flag pattern),
    so sharing them between instances without copying is safe. Series and
    loop-site child slots hold ``None`` placeholders here.
``series``
    ``(slot, max_bars_back)`` pairs; :func:`_make_state` puts a fresh
    :class:`~pynecore.core.series.SeriesImpl` into these slots.
``varip``
    Slot indexes of ``varip`` variables (excluded from var rollback).
``children``
    ``(slot, call_id, in_loop)`` triples describing the isolated call sites
    of the scope. Straight-line sites start as ``None`` and are filled by
    :func:`__resolve_slot__` on first call; loop sites hold a list of child
    states indexed by the per-invocation call counter and grown by
    :func:`__grow__`.
``names``
    Optional tuple of per-slot debug names (same order as ``init``); used
    only by :func:`explain_state` and the dump display-rewrite.

Call shapes emitted by the transformer:

- fast path, straight-line site::

    ema((__st__ if (__st__ := __state__[5]) is not None
         else __resolve_slot__(__state__, 5, ema)), close, 12)

- fast path, loop site (with the per-invocation counter ``__cnt_0__``)::

    ema((__sl0__[__i__] if (__i__ := (__cnt_0__ := __cnt_0__ + 1) - 1) < len(__sl0__)
         else __grow__(__sl0__, ema)), x, 12)

- uniform path (callee unknown at transform time), anchored at slot 7::

    (__b__[1] if (__b__ := __state__[7]) is not None and __b__[0] is f
     else __bind_any__(__state__, 7, f))(x)

- uniform path in a loop (anchor slot holds a list of ``(callee, bound)``
  pairs indexed by the per-invocation counter, so every iteration keeps its
  own instance, like the legacy counter-keyed cache did)::

    (__b__[1] if (__i__ := (__cnt_0__ := __cnt_0__ + 1) - 1) < len(__chl_0__)
     and (__b__ := __chl_0__[__i__])[0] is f
     else __bind_any_loop__(__chl_0__, __i__, f))(x)

Semantics note: when the callee at a uniform site genuinely changes (``g = a
if c else b; g(x)``), the identity check misses and the site is rebound with
FRESH state. State does not survive an a -> b -> a swap; the legacy scheme did
not support that either (a cache hit there reused the first callee's instance
regardless of the current value). A miss caused merely by a per-bar
redefinition of the SAME logical callee (a method/function nested in ``main``
is a new object every bar) is NOT a change: the rebind reuses the prior state
vector (matched by the module-level layout object), so the callee's series /
var / varip slots survive across bars — see :func:`_carry_state`.
"""
from typing import Any, Callable, Iterable
from copy import copy, deepcopy
from dataclasses import replace as dataclass_replace
from functools import partial

from .pine_export import Exported
from .series import SeriesImpl

__all__ = [
    '__resolve_slot__', '__grow__', '__bind_any__', '__bind_any_loop__',
    '__attach_layout__', '__dyn_default__',
    'create_root', 'get_root', 'discard_root', 'reset', 'register_shared_cache',
    'RootVarSnapshot', 'RootSeriesSnapshot', 'explain_state',
]

# Sentinel for dynamic parameter defaults (DynamicDefaultTransformer). A
# default referencing per-bar runtime state (``lib.hl2`` etc.) must be
# evaluated per CALL, not at ``def`` time: an anchored call site binds the
# callee closure ONCE (an ``Exported`` proxy keeps a stable identity across
# per-bar redefinitions), so a def-time default would freeze the first bar's
# value. The transformer replaces such defaults with this sentinel and
# evaluates the original expression in the function body when the argument
# was omitted.
__dyn_default__ = object()

# Root state vectors by key; only roots are registered globally, every other
# instance lives in the tree hanging off them.
_root_vectors: dict[str, tuple[list, dict[str, Any]]] = {}

# Module-lifetime bound caches of the anchorless fallbacks (an overload
# dispatcher's own cache, method_call's per-method cache). They live outside
# the root-vector tree, so reset() clears them explicitly.
_shared_caches: list[dict] = []


def register_shared_cache(cache: dict) -> dict:
    """Register a module-lifetime bound cache for clearing on :func:`reset`.

    Anchorless call paths (direct dispatcher calls, ``method_call`` dispatch)
    keep their bound instances in module-lifetime dicts instead of anchor
    slots. The legacy runtime kept such state in its global instance cache,
    which ``reset()`` dropped between runs — registering the dict keeps that
    contract.

    :param cache: The cache dict (held by reference, never replaced).
    :return: The same dict, for inline registration at the definition site.
    """
    _shared_caches.append(cache)
    return cache


def _make_state(layout: dict[str, Any]) -> list:
    """Instantiate a state vector from a layout entry.

    Template values are immutable by construction, so a flat ``list(init)``
    needs no copying; the mutable content (series buffers, loop-site child
    lists) is created fresh here.

    :param layout: Layout entry (see module docstring).
    :return: New state vector.
    """
    state = list(layout['init'])
    for slot, max_bars_back in layout['series']:
        state[slot] = SeriesImpl(max_bars_back)
    for slot, _call_id, in_loop in layout['children']:
        if in_loop:
            state[slot] = []
    return state


def __resolve_slot__(parent: list, slot: int, func: Any) -> list:
    """Cold path of a straight-line fast-path call site: create the child
    state and park it in the parent's slot.

    :param parent: The caller's state vector.
    :param slot: Child slot index assigned at transform time.
    :param func: The state-carrying callee (carries ``__pyne_layout__``).
    :return: The new child state vector.
    """
    state = _make_state(func.__pyne_layout__)
    parent[slot] = state
    return state


def __grow__(children: list, func: Any) -> list:
    """Cold path of a loop-shaped fast-path call site: append a fresh child
    state for a new loop iteration.

    :param children: The child list living in the parent's slot.
    :param func: The state-carrying callee (carries ``__pyne_layout__``).
    :return: The new child state vector.
    """
    state = _make_state(func.__pyne_layout__)
    children.append(state)
    return state


def __attach_layout__(layout: dict[str, Any]) -> Callable[[Callable], Callable]:
    """Decorator form of the layout attach, emitted for DECORATED
    state-carrying definitions. It sits in the innermost decorator position,
    so it tags the raw function before any other decorator (``overload`` in
    particular) wraps or replaces it — the post-definition
    ``func.__pyne_layout__ = ...`` assignment would tag the decorator's
    return value instead.

    :param layout: The function's layout entry.
    :return: Identity decorator that attaches the layout.
    """
    def attach(func: Any) -> Callable:
        func.__pyne_layout__ = layout
        return func
    return attach


def _carry_state(prev: tuple | None, layout: dict[str, Any]) -> list:
    """State vector for a state-carrying callee at an anchored site: reuse the
    prior anchor's vector when it belongs to the SAME logical callee, else make
    a fresh one.

    An identity miss at a uniform site has two causes that must not be
    conflated. A genuinely different callee (``g = a if c else b; g(x)``) must
    get fresh state. But a method/function nested in a per-bar ``main`` is a
    BRAND-NEW function object every bar while remaining the same logical
    callee, so its anchor also misses every bar — and there its series / var /
    varip slots must SURVIVE, not reset. The discriminator is the module-level
    layout object: it is the same dict for the same scope across bars and a
    distinct dict for every other scope, so ``prev``'s layout being the new
    callee's layout means "same callee, redefined" -> keep the state vector,
    take the closure from the new object. This is the split
    :func:`pine_method._bound_method` and ``overload._anchored`` already use;
    a real ``a -> b -> a`` swap still loses state (distinct layouts), matching
    the documented uniform-site semantics.

    :param prev: The ``(callee, bound)`` pair previously parked in the anchor
        slot, or ``None`` on the first bind.
    :param layout: The new callee's layout entry.
    :return: The state vector to bind.
    """
    if prev is not None:
        prev_bound = prev[1]
        if type(prev_bound) is partial and prev_bound.args \
                and getattr(prev_bound.func, '__pyne_layout__', None) is layout:
            return prev_bound.args[0]
    return _make_state(layout)


def _bind_target(func: Any, prev: tuple | None = None) -> Callable:
    """Binding logic of the uniform path: the legacy per-call entry guards
    (type, classmethod, Exported unwrap) run here, once per binding, not per
    call; state-carrying callees get a state vector baked into a partial,
    reused from ``prev`` across a per-bar redefinition (see :func:`_carry_state`).

    Callees that publish a ``__pyne_bind__`` factory (overload dispatchers)
    get a fresh per-anchor binding from it — that is how the dispatcher
    receives the caller's anchor and keeps one instance per implementation
    in it.

    :param func: The callee as it appears at the call site.
    :param prev: The anchor's previous ``(callee, bound)`` entry, if any.
    :return: The bound callable to invoke.
    """
    target = func
    if isinstance(target, Exported):
        target = target.__fn__
        if target is None:
            raise ValueError("Exported proxy has not been initialized with a function yet")
    bind = getattr(target, '__pyne_bind__', None)
    if bind is not None:
        return bind()
    if isinstance(target, type) or (
            hasattr(target, '__self__') and isinstance(target.__self__, type)):
        return target
    layout = getattr(target, '__pyne_layout__', None)
    return partial(target, _carry_state(prev, layout)) if layout is not None else target


def __bind_any__(parent: list, slot: int, func: Any) -> Callable:
    """Bind a callee of unknown layout at an anchored call site (uniform
    path).

    The anchor key is the ORIGINAL call-site value (e.g. the ``Exported``
    proxy itself), never the unwrapped function — the hot-path identity
    check compares against the call-site value. A state-carrying callee
    redefined for a new bar keeps its prior state vector (see
    :func:`_carry_state`).

    :param parent: The caller's state vector.
    :param slot: Anchor slot index assigned at transform time.
    :param func: The callee as it appears at the call site.
    :return: The bound callable to invoke.
    """
    bound = _bind_target(func, parent[slot])
    parent[slot] = (func, bound)
    return bound


def __bind_any_loop__(children: list, index: int, func: Any) -> Callable:
    """Bind a callee at a loop-shaped anchored call site: the anchor slot
    holds a list of ``(callee, bound)`` pairs indexed by the per-invocation
    counter, so each iteration keeps its own instance. Rebinds in place on
    an identity miss, reusing the iteration's prior state vector when the same
    logical callee was redefined for a new bar (see :func:`_carry_state`).

    :param children: The pair list living in the parent's anchor slot.
    :param index: Current iteration index (counter is sequential, so the
        grow case is always ``index == len(children)``).
    :param func: The callee as it appears at the call site.
    :return: The bound callable to invoke.
    """
    prev = children[index] if index < len(children) else None
    bound = _bind_target(func, prev)
    entry = (func, bound)
    if index < len(children):
        children[index] = entry
    else:
        children.append(entry)
    return bound


def create_root(key: str, layout: dict[str, Any]) -> list:
    """Create (or recreate) a root state vector.

    Roots belong to the entry points the runner drives directly: the script
    ``main()``, library mains and security-process entries. Recreating an
    existing key replaces the old root (a rerun drops the previous tree).

    :param key: Unique root key (e.g. the module path of the entry point).
    :param layout: Layout entry of the root scope.
    :return: The new root state vector.
    """
    state = _make_state(layout)
    _root_vectors[key] = (state, layout)
    return state


def get_root(key: str) -> list | None:
    """Return a registered root state vector, or ``None``.

    :param key: Root key used at :func:`create_root`.
    :return: The root state vector if registered.
    """
    entry = _root_vectors.get(key)
    return entry[0] if entry is not None else None


def discard_root(key: str) -> None:
    """Drop a root vector (its tree dies through GC). Missing keys are ignored.

    :param key: Root key used at :func:`create_root`.
    """
    _root_vectors.pop(key, None)


def reset() -> None:
    """Drop every function instance: clear the child slots of all root
    vectors and the registered module-lifetime bound caches. Var and series
    slots of the roots are left untouched — exact parity with the legacy
    ``function_isolation.reset()``, which cleared the instance cache but
    never touched main's own state.
    """
    for state, layout in _root_vectors.values():
        for slot, _call_id, in_loop in layout['children']:
            state[slot] = [] if in_loop else None
    for cache in _shared_caches:
        cache.clear()


def _var_slots(layout: dict[str, Any]) -> tuple[int, ...]:
    """Slots subject to var rollback: everything that is not a series, varip
    or child slot.

    :param layout: Layout entry.
    :return: Rollback slot indexes.
    """
    excluded = {slot for slot, _max_bars_back in layout['series']}
    excluded.update(layout['varip'])
    excluded.update(slot for slot, _call_id, _in_loop in layout['children'])
    return tuple(i for i in range(len(layout['init'])) if i not in excluded)


def _copy_value(value: Any) -> Any:
    """Copy a value for snapshot/restore: immutables as-is, dicts/lists by
    deepcopy, dataclasses by ``replace``, everything else by shallow copy.

    :param value: Value to copy.
    :return: Copied (or immutable, as-is) value.
    """
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, (dict, list)):
        return deepcopy(value)
    try:
        return dataclass_replace(value)  # type: ignore[type-var]
    except TypeError:
        return copy(value)


class RootVarSnapshot:
    """Snapshot/restore of the ``var`` slots of the root vectors, for the
    calc_on_order_fills rollback. Parity with the legacy ``VarSnapshot``:
    varip slots are excluded and isolated child instances are not touched.

    Passing ``keys`` scopes the snapshot to specific roots — the runner uses
    its own root keys, so interleaved runner instances never roll back each
    other's state (the legacy snapshot was scoped to explicit modules).
    """

    __slots__ = ('_targets', '_snapshots')

    def __init__(self, keys: Iterable[str] | None = None):
        self._targets: list[tuple[list, tuple[int, ...]]] = []
        self._snapshots: list[list] = []
        entries = (_root_vectors.values() if keys is None
                   else (_root_vectors[key] for key in keys if key in _root_vectors))
        for state, layout in entries:
            slots = _var_slots(layout)
            if slots:
                self._targets.append((state, slots))

    @property
    def has_vars(self) -> bool:
        """Whether any root has var slots to roll back."""
        return bool(self._targets)

    def save(self) -> None:
        """Snapshot the var slots of all roots (called at bar start)."""
        self._snapshots = [[_copy_value(state[i]) for i in slots]
                           for state, slots in self._targets]

    def restore(self) -> None:
        """Restore the var slots of all roots to the saved snapshot."""
        for (state, slots), snapshot in zip(self._targets, self._snapshots):
            for i, value in zip(slots, snapshot):
                state[i] = _copy_value(value)


# noinspection PyProtectedMember
class RootSeriesSnapshot:
    """Snapshot/restore of the ``series`` slots of the root vectors.

    Companion to :class:`RootVarSnapshot` for the live
    ``request.security_lower_tf`` LTF baseline. A reordered feed can force the
    collector to replay an *earlier* ``bar_index`` after a later one already ran;
    since :meth:`SeriesImpl.add` only overwrites for the current ``bar_index``,
    that backward re-run would append and grow the buffer. ``RootVarSnapshot``
    deliberately excludes series slots, so they need their own rollback.

    Only the ROOT series slots are captured: a builtin price series like
    ``close`` (the backing of ``close[1]``) is anchored in ``main`` by
    ``LibrarySeriesTransformer``, so it lives in a root series slot. Child
    (function-instance) series are dropped by :func:`reset` before every replay
    and re-created fresh, so they never carry a backward-append across a replay
    and need no snapshot here.
    """

    __slots__ = ('_targets', '_snapshots')

    def __init__(self, keys: Iterable[str] | None = None):
        self._targets: list[tuple[list, tuple[int, ...]]] = []
        self._snapshots: list[list] = []
        entries = (_root_vectors.values() if keys is None
                   else (_root_vectors[key] for key in keys if key in _root_vectors))
        for state, layout in entries:
            slots = tuple(slot for slot, _max_bars_back in layout['series'])
            if slots:
                self._targets.append((state, slots))

    @property
    def has_series(self) -> bool:
        """Whether any root has series slots to roll back."""
        return bool(self._targets)

    @property
    def saved(self) -> bool:
        """Whether a snapshot has been captured (``save`` called since init)."""
        return bool(self._snapshots)

    def save(self) -> None:
        """Snapshot the buffer state of every root series slot."""
        self._snapshots = [[state[i]._snapshot() for i in slots]
                           for state, slots in self._targets]

    def restore(self) -> None:
        """Restore every root series slot to the saved snapshot (in place)."""
        for (state, slots), snapshot in zip(self._targets, self._snapshots):
            for i, snap in zip(slots, snapshot):
                state[i]._restore(snap)


def explain_state(func_or_layout: Any, state: list) -> dict[str, Any]:
    """Render a state vector as a readable name -> value dict (debug helper;
    callable from a debugger watch window).

    :param func_or_layout: A state-carrying function (``__pyne_layout__`` is
        read off it) or a layout entry itself.
    :param state: The instance's state vector.
    :return: Slot name (or descriptive fallback label) -> current value.
    """
    layout: dict[str, Any] = getattr(func_or_layout, '__pyne_layout__', func_or_layout)
    names = layout.get('names')
    series_slots = {slot for slot, _max_bars_back in layout['series']}
    child_ids = {slot: call_id for slot, call_id, _in_loop in layout['children']}
    out: dict[str, Any] = {}
    for i, value in enumerate(state):
        if names and i < len(names) and names[i]:
            label = names[i]
        elif i in child_ids:
            label = f'slot_{i}·child·{child_ids[i]}'
        elif i in series_slots:
            label = f'slot_{i}·series'
        else:
            label = f'slot_{i}'
        out[label] = value
    return out
