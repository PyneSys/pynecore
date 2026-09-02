"""
Unit tests for the slot-based instance state runtime (core/instance_state.py).
"""
import sys
from pathlib import Path

import pytest

from pynecore.core.instance_state import (
    __resolve_slot__, __bind_any__, __loop_state__, __bind_loop__,
    __attach_layout__,
    create_root, get_root, discard_root, reset, register_shared_cache,
    RootVarSnapshot, explain_state, _make_state,
)
from pynecore.core.pine_export import Exported
from pynecore.core.series import SeriesImpl

LAYOUT_LEAF = {
    'init': (0.0,),
    'series': (),
    'varip': (),
    'children': (),
}

# slot 0: var, slot 1: series, slot 2: straight-line child,
# slot 3: loop-site cell, slot 4: varip
LAYOUT_PARENT = {
    'init': (0, None, None, None, False),
    'series': ((1, 10, None),),
    'varip': (4,),
    'children': ((2, 'main·acc·0', False), (3, 'main·acc·1', True)),
    'names': ('count', 'src', 'acc·0', 'acc·1', 'flag'),
}

# A builtin machine's layout: the ``compacted`` flag (the ``@pyne lib``
# marker) is what makes it a same-bar rollback target at loop sites
LAYOUT_BUILTIN = {
    'init': (0.0,),
    'series': (),
    'varip': (),
    'children': (),
    'compacted': True,
}


def _set_bar(index: int) -> None:
    """Drive the runner-maintained global bar index the loop helpers read."""
    from pynecore import lib
    lib.bar_index = index

# Two plain var slots: a drawing held directly and one held in a container
LAYOUT_DRAWINGS = {
    'init': (None, None),
    'series': (),
    'varip': (),
    'children': (),
}


def _make_stateful(layout=LAYOUT_LEAF):
    """Create a state-carrying callee the way the transformer would emit it."""
    def acc(__state__, x):
        __state__[0] = __state__[0] + x
        return __state__[0]
    acc.__pyne_layout__ = layout
    return acc


def __test_make_state__():
    """ _make_state: shared immutables, fresh series, empty child slots """
    s1 = _make_state(LAYOUT_PARENT)
    s2 = _make_state(LAYOUT_PARENT)
    assert s1[:-1] == [0, s1[1], None, None, False]
    assert s1[-1] is LAYOUT_PARENT  # trailing layout reference (see _make_state)
    assert isinstance(s1[1], SeriesImpl)
    assert s1[1] is not s2[1]


def __test_resolve_slot__():
    """ __resolve_slot__: fills the parent slot, instances are independent """
    acc = _make_stateful()
    parent = _make_state(LAYOUT_PARENT)
    child = __resolve_slot__(parent, 2, acc)
    assert parent[2] is child
    assert acc(child, 1.0) == 1.0
    assert acc(child, 2.0) == 3.0
    # the same callee resolved into another parent starts fresh
    other = __resolve_slot__(_make_state(LAYOUT_PARENT), 2, acc)
    assert acc(other, 5.0) == 5.0
    assert child[0] == 3.0


def __test_loop_state_shared_instance__():
    """ __loop_state__: ONE instance shared by every iteration; user state
    accumulates across the same-bar calls and across bars """
    acc = _make_stateful()
    parent = _make_state(LAYOUT_PARENT)
    _set_bar(0)
    first = __loop_state__(parent, 3, acc)
    second = __loop_state__(parent, 3, acc)
    assert first is second
    assert parent[3][0] is first
    assert acc(first, 1.0) == 1.0
    assert acc(__loop_state__(parent, 3, acc), 1.0) == 2.0
    _set_bar(1)
    assert acc(__loop_state__(parent, 3, acc), 1.0) == 3.0


def __test_loop_state_builtin_rollback__():
    """ __loop_state__: a builtin machine (compacted layout) re-derives every
    same-bar call from bar-start state; a new bar commits the last result;
    a ``per_call`` machine keeps advancing across iterations """
    machine = _make_stateful(LAYOUT_BUILTIN)
    parent = _make_state(LAYOUT_PARENT)
    _set_bar(0)
    assert machine(__loop_state__(parent, 3, machine), 1.0) == 1.0
    # Same bar: rolled back to the 0.0 bar-start state, not 1.0 + 2.0
    assert machine(__loop_state__(parent, 3, machine), 2.0) == 2.0
    _set_bar(1)
    # New bar: the bar's LAST result (2.0) is the committed state
    assert machine(__loop_state__(parent, 3, machine), 1.0) == 3.0

    percall = _make_stateful({**LAYOUT_BUILTIN, 'per_call': True})
    parent2 = _make_state(LAYOUT_PARENT)
    _set_bar(0)
    assert percall(__loop_state__(parent2, 3, percall), 1.0) == 1.0
    assert percall(__loop_state__(parent2, 3, percall), 1.0) == 2.0


def __test_loop_state_midbar_builtin_reinit__():
    """ __loop_state__: a builtin child created MID-BAR (inside an earlier
    iteration, missing from the bar-start snapshot) is re-initialized on the
    same-bar rollback — its bar-start state is "not existing yet" """
    machine = _make_stateful(LAYOUT_BUILTIN)

    def wrapper(__state__, x):
        __state__[0] += 1  # user var: accumulates across iterations
        child = __state__[2]
        if child is None:
            child = __resolve_slot__(__state__, 2, machine)
        return __state__[0], machine(child, x)
    wrapper.__pyne_layout__ = LAYOUT_PARENT

    parent = _make_state(LAYOUT_PARENT)
    _set_bar(0)
    assert wrapper(__loop_state__(parent, 3, wrapper), 1.0) == (1, 1.0)
    # Same bar: the var went on, the machine was re-initialized (1.0, not 2.0)
    assert wrapper(__loop_state__(parent, 3, wrapper), 1.0) == (2, 1.0)
    _set_bar(1)
    # New bar: the machine's committed state is the bar's last result
    assert wrapper(__loop_state__(parent, 3, wrapper), 1.0) == (3, 2.0)


def __test_bind_any_stateful__():
    """ __bind_any__: state-carrying callee gets a bound partial, state persists;
    a rebind keeps the state vector when the same logical callee was redefined
    for a new bar (same layout), and resets only on a genuinely different
    callee (distinct layout) """
    acc = _make_stateful()
    parent = _make_state(LAYOUT_PARENT)
    bound = __bind_any__(parent, 2, acc)
    assert parent[2] == (acc, bound)
    assert bound(1.0) == 1.0
    assert bound(2.0) == 3.0
    # Same logical callee redefined for a new bar (new object, same layout):
    # the identity check misses, but the state vector is kept.
    rebound = __bind_any__(parent, 2, _make_stateful())
    assert rebound(1.0) == 4.0  # continues the slot's state (3.0 + 1.0)
    # A genuinely different callee (distinct layout) gets fresh state.
    other = __bind_any__(parent, 2, _make_stateful(layout=dict(LAYOUT_LEAF)))
    assert other(1.0) == 1.0


def __test_bind_any_stateless__():
    """ __bind_any__: plain callables and classes are bound as-is """
    def plain(x):
        return x + 1

    parent = _make_state(LAYOUT_PARENT)
    assert __bind_any__(parent, 2, plain) is plain
    assert parent[2] == (plain, plain)
    assert __bind_any__(parent, 2, SeriesImpl) is SeriesImpl


def __test_bind_any_exported__():
    """ __bind_any__: Exported is unwrapped, the anchor key is the proxy """
    acc = _make_stateful()
    exported = Exported()
    exported.set(acc)
    parent = _make_state(LAYOUT_PARENT)
    bound = __bind_any__(parent, 2, exported)
    assert parent[2][0] is exported
    assert bound(2.0) == 2.0
    assert bound(3.0) == 5.0
    with pytest.raises(ValueError):
        __bind_any__(parent, 2, Exported())


def __test_attach_layout__():
    """ __attach_layout__: tags the raw function and returns it unchanged """
    @__attach_layout__(LAYOUT_LEAF)
    def acc(__state__, x):
        __state__[0] += x
        return __state__[0]

    assert acc.__pyne_layout__ is LAYOUT_LEAF
    state = _make_state(LAYOUT_LEAF)
    assert acc(state, 2.0) == 2.0


def __test_bind_any_dispatcher_hook__():
    """ __bind_any__: a callee with __pyne_bind__ gets a fresh per-anchor
    binding from the factory, and the call site's overload pin reaches it """
    seen = []

    def fake_dispatcher():
        raise AssertionError("the anchor must call the factory's binding")

    def bound_instance(x):
        return x * 10

    def factory(pin=None):
        seen.append(pin)
        return bound_instance
    fake_dispatcher.__pyne_bind__ = factory

    parent = _make_state(LAYOUT_PARENT)
    bound = __bind_any__(parent, 2, fake_dispatcher)
    assert bound is bound_instance
    assert parent[2] == (fake_dispatcher, bound_instance)
    assert bound(2) == 20
    assert seen == [None]  # an unpinned site still binds

    parent = _make_state(LAYOUT_PARENT)
    __bind_any__(parent, 2, fake_dispatcher, 'if')
    assert seen == [None, 'if']


def __test_bind_loop__():
    """ __bind_loop__: ONE shared instance across iterations; an in-place
    rebind keeps the state when the same logical callee was redefined (same
    layout), and resets only on a genuinely different callee (distinct
    layout) """
    acc = _make_stateful()
    parent = _make_state(LAYOUT_PARENT)
    _set_bar(0)
    first = __bind_loop__(parent, 3, acc)
    assert __bind_loop__(parent, 3, acc) is first
    assert parent[3][0] == (acc, first)
    assert first(1.0) == 1.0
    assert first(2.0) == 3.0  # user state accumulates across iterations
    # Same logical callee redefined for a new bar (new object, same layout):
    # the identity check misses, but the shared state vector is kept.
    _set_bar(1)
    redefined = _make_stateful()
    rebound = __bind_loop__(parent, 3, redefined)
    assert parent[3][0] == (redefined, rebound)
    assert rebound(1.0) == 4.0  # continues the cell's state (3.0 + 1.0)
    # A genuinely different callee (distinct layout) resets to fresh state.
    other = __bind_loop__(parent, 3, _make_stateful(layout=dict(LAYOUT_LEAF)))
    assert other(1.0) == 1.0


def __test_reset__():
    """ reset(): clears child slots of roots, leaves var/series slots alone """
    acc = _make_stateful()
    root = create_root('test·reset', LAYOUT_PARENT)
    try:
        __resolve_slot__(root, 2, acc)
        _set_bar(0)
        __loop_state__(root, 3, acc)
        root[0] = 42
        series = root[1]
        reset()
        assert root[2] is None
        assert root[3] is None
        assert root[0] == 42
        assert root[1] is series
        assert get_root('test·reset') is root
    finally:
        discard_root('test·reset')
    assert get_root('test·reset') is None


def __test_register_shared_cache__():
    """ reset(): registered module-lifetime bound caches are cleared """
    cache = register_shared_cache({'bound': object()})
    assert cache
    reset()
    assert cache == {}


def __test_root_var_snapshot__():
    """ RootVarSnapshot: var slots roll back, varip/series/children do not """
    acc = _make_stateful()
    root = create_root('test·snap', LAYOUT_PARENT)
    try:
        root[0] = 1
        root[4] = False
        snapshot = RootVarSnapshot()
        assert snapshot.has_vars
        snapshot.save()
        child = __resolve_slot__(root, 2, acc)
        series = root[1]
        root[0] = 99
        root[4] = True
        snapshot.restore()
        assert root[0] == 1
        assert root[4] is True       # varip survives the rollback
        assert root[2] is child      # children are not touched
        assert root[1] is series     # series objects are not touched
    finally:
        discard_root('test·snap')


def __test_root_var_snapshot_keeps_drawing_handles__():
    """ A var-held drawing rolls back as the same registered object, not as a clone """
    from pynecore.core import viz
    from pynecore.lib import array, label

    viz.reset_state()
    root = create_root('test·snap·draw', LAYOUT_DRAWINGS)
    try:
        drawing = label.new(1, 10.0, "A")
        root[0] = drawing
        root[1] = array.new_label()
        array.push(root[1], drawing)

        snapshot = RootVarSnapshot(['test·snap·draw'])
        snapshot.save()
        root[0] = label.new(2, 20.0, "B")
        array.clear(root[1])
        snapshot.restore()

        # A field-wise clone carries the SAME vid, sits in no registry and never
        # reaches the chart, so the script would go on mutating an invisible object
        # while the registered original went unreferenced.
        assert root[0] is drawing
        assert drawing in label._registry
        # The container itself rolls back, but its elements stay handles.
        assert len(root[1]) == 1
        assert root[1][0] is drawing
    finally:
        discard_root('test·snap·draw')
        viz.reset_state()


def __test_root_var_snapshot_keys__():
    """ RootVarSnapshot(keys): only the named roots are covered """
    mine = create_root('test·snap·mine', LAYOUT_PARENT)
    other = create_root('test·snap·other', LAYOUT_PARENT)
    try:
        mine[0] = 1
        other[0] = 1
        snapshot = RootVarSnapshot(['test·snap·mine', 'test·snap·gone'])
        assert snapshot.has_vars
        snapshot.save()
        mine[0] = 99
        other[0] = 99
        snapshot.restore()
        assert mine[0] == 1
        assert other[0] == 99  # foreign root is not rolled back
    finally:
        discard_root('test·snap·mine')
        discard_root('test·snap·other')


def __test_varip_add_assign_rollback_exclusion__():
    """ A varip accumulator stays out of the var rollback, the plain one does not """
    from pynecore.core.script_runner import import_script

    # A real transformed script, so the transformer-side flag inheritance is
    # covered too, not just the runtime exclusion
    module = import_script(Path(__file__).parent / 'data' / 'varip_add_assign.py')
    try:
        layout = module.main.__pyne_layout__
        names = layout['names']
        varip_slots = set(layout['varip'])
        assert names.index('varip_total') in varip_slots
        assert names.index('plain') not in varip_slots

        # Runtime: a rollback restores the plain slot, skips the varip one
        root = create_root('test·varip·add', layout)
        try:
            snapshot = RootVarSnapshot(['test·varip·add'])
            snapshot.save()
            for slot in range(len(names)):
                root[slot] = 99.0
            snapshot.restore()
            assert root[names.index('plain')] == 0.0
            assert root[names.index('varip_total')] == 99.0
        finally:
            discard_root('test·varip·add')
    finally:
        sys.modules.pop('varip_add_assign', None)


def __test_explain_state__():
    """ explain_state: named slots map to values, fallback labels are descriptive """
    acc = _make_stateful(LAYOUT_PARENT)
    state = _make_state(LAYOUT_PARENT)
    state[0] = 7
    named = explain_state(acc, state)
    assert named['count'] == 7
    assert named['acc·0'] is None
    unnamed_layout = dict(LAYOUT_PARENT)
    del unnamed_layout['names']
    unnamed = explain_state(unnamed_layout, state)
    assert unnamed['slot_0'] == 7
    assert unnamed['slot_2·child·main·acc·0'] is None
    assert 'slot_1·series' in unnamed
