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
    ``(slot, max_bars_back, elem)`` triples; :func:`_make_state` puts a fresh
    :class:`~pynecore.core.series.SeriesImpl` into these slots. ``elem`` is
    the statically known element type name (``'float'`` selects the native
    nan as the series' out-of-range na value) or ``None``.
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
``compacted``
    Present and true only for ``@pyne lib`` modules: their series are the
    rolling windows of the builtin machines, which skip na bars on purpose,
    so the per-bar forward fill of :meth:`SeriesImpl.add` must stay off.

Call shapes emitted by the transformer:

- fast path, straight-line site::

    ema((__st·__ if (__st·__ := __state__[5]) is not None
         else __resolve_slot·__(__state__, 5, ema)), close, 12)

- fast path, loop site (with the per-invocation counter ``__cnt·0__``)::

    ema((__chl·0__[__i·__] if (__i·__ := (__cnt·0__ := __cnt·0__ + 1) - 1) < len(__chl·0__)
         else __grow·__(__chl·0__, ema)), x, 12)

- uniform path (callee unknown at transform time), anchored at slot 7::

    (__b·__[1] if (__b·__ := __state__[7]) is not None and __b·__[0] is f
     else __bind_any·__(__state__, 7, f))(x)

- uniform path in a loop (anchor slot holds a list of ``(callee, bound)``
  pairs indexed by the per-invocation counter, so every iteration keeps its
  own instance, like the legacy counter-keyed cache did)::

    (__b·__[1] if (__i·__ := (__cnt·0__ := __cnt·0__ + 1) - 1) < len(__chl·0__)
     and (__b·__ := __chl·0__[__i·__])[0] is f
     else __bind_any_loop·__(__chl·0__, __i·__, f))(x)

- uniform path whose CALLEE EXPRESSION runs code (a call hides in it, e.g.
  ``bump().upper()``): the callee is bound once in a leading tautological
  conjunct, so it is evaluated exactly once and before the anchor read::

    (__b·__[1] if (__c·__ := bump(__state__[0]).upper) is __c·__
     and (__b·__ := __state__[1]) is not None and __b·__[0] is __c·__
     else __bind_any·__(__state__, 1, __c·__))()

  The conjunct must stay first and must not be falsifiable: an inner call site
  writes ``__b·__`` and the loop counter, and a short circuit would skip them.

- comprehension ITERABLE positions, where Python rejects every assignment
  expression: the guard folds into a single helper call instead, and the
  loop-shaped variants advance a one-element counter cell::

    [x for x in __bind_slot·__(__state__, 7, f)()]
    [y for a in items for y in bump(__next_state·__(__chl·0__, __cnt·0__, bump), a)]

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
from typing import Any, Callable, Iterable, cast
from copy import copy, deepcopy
from dataclasses import replace as dataclass_replace
from functools import partial

from .pine_export import Exported
from .series import SeriesImpl
from ..types.base import Drawing
from ..types.na import na_float

__all__ = [
    '__resolve_slot__', '__grow__', '__bind_any__', '__bind_any_loop__',
    '__slot_state__', '__next_state__', '__bind_slot__', '__bind_next__',
    '__attach_layout__', '__dyn_default__',
    'create_root', 'get_root', 'discard_root', 'reset', 'register_shared_cache',
    'RootVarSnapshot', 'RootSeriesSnapshot', 'RootChildSnapshot', 'explain_state',
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
    compacted = layout.get('compacted', False)
    for slot, max_bars_back, elem in layout['series']:
        state[slot] = SeriesImpl(max_bars_back, na_float if elem == 'float' else None, compacted)
    for slot, _call_id, in_loop in layout['children']:
        if in_loop:
            state[slot] = []
    # Trailing layout reference: slot addressing uses literal non-negative
    # indexes only, so the extra element is invisible to emitted code. It lets
    # a walker that meets a bare child state vector (fast-path slots hold the
    # list with no callee at hand) recover the vector's layout — the child
    # snapshot of the calc_on_order_fills rollback needs exactly that.
    state.append(layout)
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


def __slot_state__(parent: list, slot: int, func: Any) -> list:
    """Whole straight-line fast-path call site, guard included.

    Emitted where the inline guard cannot be: Python rejects an assignment
    expression anywhere inside a comprehension's iterable expression, so the
    ``__st·__`` walrus form is not available there.

    :param parent: The caller's state vector.
    :param slot: Child slot index assigned at transform time.
    :param func: The state-carrying callee (carries ``__pyne_layout__``).
    :return: The child state vector.
    """
    state = parent[slot]
    return state if state is not None else __resolve_slot__(parent, slot, func)


def __next_state__(children: list, counter: list, func: Any) -> list:
    """Whole loop-shaped fast-path call site, counter advance included.

    The per-invocation counter lives in a one-element list here instead of a
    plain local, because the walrus that would advance a local is illegal in
    the comprehension-iterable position this form serves.

    :param children: The child list living in the parent's slot.
    :param counter: One-element counter cell, reset in the function prologue.
    :param func: The state-carrying callee (carries ``__pyne_layout__``).
    :return: This iteration's child state vector.
    """
    index = counter[0]
    counter[0] = index + 1
    return children[index] if index < len(children) else __grow__(children, func)


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
    target: Any = func
    if isinstance(target, Exported):
        unwrapped = target.__fn__
        if unwrapped is None:
            raise ValueError("Exported proxy has not been initialized with a function yet")
        target = unwrapped
    bind = getattr(target, '__pyne_bind__', None)
    if bind is not None:
        return bind()
    if isinstance(target, type) or (
            hasattr(target, '__self__') and isinstance(target.__self__, type)):
        return target
    layout: dict[str, Any] | None = getattr(target, '__pyne_layout__', None)
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


def __bind_slot__(parent: list, slot: int, func: Any) -> Callable:
    """Whole straight-line anchored call site, identity check included.

    Emitted where the inline guard cannot be (comprehension iterable
    expressions, see :func:`__slot_state__`). The callee expression is
    evaluated exactly once — as this call's argument — so a callee that runs
    code needs no temporary here.

    :param parent: The caller's state vector.
    :param slot: Anchor slot index assigned at transform time.
    :param func: The callee as it appears at the call site.
    :return: The bound callable to invoke.
    """
    pair = parent[slot]
    if pair is not None and pair[0] is func:
        return pair[1]
    return __bind_any__(parent, slot, func)


def __bind_next__(children: list, counter: list, func: Any) -> Callable:
    """Whole loop-shaped anchored call site, counter advance included.

    :param children: The pair list living in the parent's anchor slot.
    :param counter: One-element counter cell, reset in the function prologue.
    :param func: The callee as it appears at the call site.
    :return: The bound callable to invoke.
    """
    index = counter[0]
    counter[0] = index + 1
    if index < len(children):
        pair = children[index]
        if pair[0] is func:
            return pair[1]
    return __bind_any_loop__(children, index, func)


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
    excluded = {slot for slot, _max_bars_back, _elem in layout['series']}
    excluded.update(layout['varip'])
    excluded.update(slot for slot, _call_id, _in_loop in layout['children'])
    return tuple(i for i in range(len(layout['init'])) if i not in excluded)


def _copy_value(value: Any) -> Any:
    """Copy a value for snapshot/restore: immutables and drawings as-is,
    dicts/lists by deepcopy, dataclasses by ``replace``, everything else by
    shallow copy.

    :param value: Value to copy.
    :return: Copied (or immutable, as-is) value.
    """
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, Drawing):
        # A drawing is a handle on a registered chart object, so the value to roll
        # back is the handle. Cloning it field by field detached the variable from
        # the registry: the script kept mutating an object with a duplicate vid that
        # never reached the chart, while the registered original went unreferenced.
        return value
    if isinstance(value, (dict, list)):
        # A container holding drawings is deep-copied too, but ``Drawing`` stops the
        # recursion at its own handles, matching TradingView's container copies.
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
            slots = tuple(slot for slot, _max_bars_back, _elem in layout['series'])
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


def _snap_vector(state: list, layout: dict[str, Any]) -> tuple:
    """Deep snapshot of one state vector and its child subtree.

    varip slots are excluded on purpose: a varip keeps its intra-bar advances
    across discarded re-executions (calc_on_order_fills, live ticks), exactly
    like TradingView's realtime rollback keeps them.
    """
    var_vals = tuple((i, _copy_value(state[i])) for i in _var_slots(layout))
    series_vals = tuple((slot, state[slot]._snapshot())
                        for slot, _max_bars_back, _elem in layout['series'])
    return var_vals, series_vals, _snap_children(state, layout)


def _snap_children(state: list, layout: dict[str, Any]) -> tuple:
    """Snapshot only the child slots of a state vector."""
    child_vals = []
    for slot, _call_id, in_loop in layout['children']:
        val = state[slot]
        if in_loop:
            child_vals.append((slot, True, tuple(_snap_child(e) for e in val)))
        else:
            child_vals.append((slot, False, _snap_child(val)))
    return tuple(child_vals)


def _snap_child(entry: Any) -> tuple:
    """Snapshot one child-slot entry.

    A bare list is a fast-path child state vector (its layout rides in the
    trailing element, see :func:`_make_state`); an anchored ``(callee, bound)``
    pair whose bound is a state-carrying partial exposes its vector as
    ``bound.args[0]``; a bound closure whose entire state is one series
    publishes it as ``__pyne_series__`` (``inline_series``). Anything else (an
    overload dispatcher's or method binder's opaque closure) cannot be walked —
    it is DROPPED on restore, which re-binds it fresh, exactly what
    :func:`reset` did for every child.
    """
    if entry is None:
        return ('none',)
    if type(entry) is list:
        return ('vec', entry, _snap_vector(entry, entry[-1]))
    if type(entry) is tuple and len(entry) == 2:
        bound = entry[1]
        if type(bound) is partial and bound.args:
            layout = getattr(bound.func, '__pyne_layout__', None)
            if layout is not None:
                return ('pair', entry, bound.args[0], _snap_vector(bound.args[0], layout))
        # A closure-held series must be ROLLED BACK, never dropped: re-binding
        # gives an empty buffer, and ``expr[n]`` then reads na forever.
        series = getattr(bound, '__pyne_series__', None)
        if series is not None:
            return ('series', entry, series, series._snapshot())
    return ('drop',)


def _restore_vector(state: list, snap: tuple) -> None:
    """Restore one state vector (in place) from its :func:`_snap_vector` form."""
    var_vals, series_vals, child_vals = snap
    for i, value in var_vals:
        state[i] = _copy_value(value)
    for slot, series_snap in series_vals:
        state[slot]._restore(series_snap)
    _restore_children(state, child_vals)


def _restore_children(state: list, child_vals: tuple) -> None:
    """Restore the child slots of a state vector from :func:`_snap_children`."""
    for slot, in_loop, child_snap in child_vals:
        if in_loop:
            lst = state[slot]
            # An element that cannot be restored in place ends the reusable
            # prefix: the per-invocation counter binds strictly sequentially,
            # so truncating there re-creates it and everything after fresh.
            keep = 0
            for saved in child_snap:
                if not _restore_child(lst, keep, saved):
                    break
                keep += 1
            del lst[keep:]
        else:
            kind = child_snap[0]
            if kind == 'none' or kind == 'drop':
                state[slot] = None
            elif kind == 'vec':
                state[slot] = child_snap[1]
                _restore_vector(child_snap[1], child_snap[2])
            elif kind == 'series':
                state[slot] = child_snap[1]
                child_snap[2]._restore(child_snap[3])
            else:  # pair
                state[slot] = child_snap[1]
                _restore_vector(child_snap[2], child_snap[3])


def _restore_child(lst: list, index: int, saved: tuple) -> bool:
    """Restore one loop-list element in place; ``False`` if it must be dropped
    (the caller truncates the list from there)."""
    kind = saved[0]
    if kind == 'vec':
        lst[index] = saved[1]
        _restore_vector(saved[1], saved[2])
        return True
    if kind == 'pair':
        lst[index] = saved[1]
        _restore_vector(saved[2], saved[3])
        return True
    if kind == 'series':
        lst[index] = saved[1]
        saved[2]._restore(saved[3])
        return True
    return False


class RootChildSnapshot:
    """Snapshot/restore of the child-instance subtrees of the root vectors,
    for discarded re-executions (calc_on_order_fills fills, live intra-bar
    ticks).

    :func:`reset` DROPS every function instance instead, which loses the
    builtin machines' internal state: ``ta.tr``'s previous close, a ``ta.rma``
    accumulator — measured on TradingView (SUPERTREND ATR corpus script,
    2026-08-13), the committed bar calculation after a fill sees the clean
    bar-start state, so the instances must be rolled back, not re-created.

    Series slots inside instances are restored too (a discarded run may have
    conditionally written them); root-level series stay untouched, matching
    :class:`RootVarSnapshot` (same-bar re-adds overwrite in place). Shared
    bound caches are cleared on restore — the entries walkable through anchor
    pairs are restored, the opaque ones re-bind fresh, both no worse than the
    :func:`reset` behavior they replace.
    """

    __slots__ = ('_roots', '_snapshots')

    def __init__(self, keys: Iterable[str] | None = None):
        self._roots: list[tuple[list, dict[str, Any]]] = []
        self._snapshots: list[tuple] = []
        entries = (_root_vectors.values() if keys is None
                   else (_root_vectors[key] for key in keys if key in _root_vectors))
        for state, layout in entries:
            self._roots.append((state, layout))

    def save(self) -> None:
        """Snapshot the child subtrees of all roots (bar-start state)."""
        self._snapshots = [_snap_children(state, layout)
                           for state, layout in self._roots]

    def restore(self) -> None:
        """Restore all roots' child subtrees to the saved snapshot."""
        for (state, _layout), snap in zip(self._roots, self._snapshots):
            _restore_children(state, snap)
        for cache in _shared_caches:
            cache.clear()


def explain_state(func_or_layout: Any, state: list) -> dict[str, Any]:
    """Render a state vector as a readable name -> value dict (debug helper;
    callable from a debugger watch window).

    :param func_or_layout: A state-carrying function (``__pyne_layout__`` is
        read off it) or a layout entry itself.
    :param state: The instance's state vector.
    :return: Slot name (or descriptive fallback label) -> current value.
    """
    layout: dict[str, Any] = cast(dict[str, Any],
                                  getattr(func_or_layout, '__pyne_layout__', func_or_layout))
    names: tuple[str, ...] | None = layout.get('names')
    series_slots = {slot for slot, _max_bars_back, _elem in layout['series']}
    child_ids = {slot: call_id for slot, call_id, _in_loop in layout['children']}
    out: dict[str, Any] = {}
    # The trailing element is the vector's layout reference, not a slot.
    for i, value in enumerate(state[:len(layout['init'])]):
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
