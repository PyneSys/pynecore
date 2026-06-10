"""
Unit tests for the slot-based instance state runtime (core/instance_state.py).
"""
import pytest

from pynecore.core.instance_state import (
    __resolve_slot__, __grow__, __bind_any__, __bind_any_loop__,
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
# slot 3: loop child list, slot 4: varip
LAYOUT_PARENT = {
    'init': (0, None, None, None, False),
    'series': ((1, 10),),
    'varip': (4,),
    'children': ((2, 'main·acc·0', False), (3, 'main·acc·1', True)),
    'names': ('count', 'src', 'acc·0', 'acc·1', 'flag'),
}


def _make_stateful(layout=LAYOUT_LEAF):
    """Create a state-carrying callee the way the transformer would emit it."""
    def acc(__state__, x):
        __state__[0] = __state__[0] + x
        return __state__[0]
    acc.__pyne_layout__ = layout
    return acc


def __test_make_state__():
    """ _make_state: shared immutables, fresh series and loop lists """
    s1 = _make_state(LAYOUT_PARENT)
    s2 = _make_state(LAYOUT_PARENT)
    assert s1 == [0, s1[1], None, [], False]
    assert isinstance(s1[1], SeriesImpl)
    assert s1[1] is not s2[1]
    assert s1[3] is not s2[3]


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


def __test_grow__():
    """ __grow__: appends one fresh child per loop iteration """
    acc = _make_stateful()
    parent = _make_state(LAYOUT_PARENT)
    first = __grow__(parent[3], acc)
    second = __grow__(parent[3], acc)
    assert parent[3] == [first, second]
    assert first is not second
    acc(first, 1.0)
    assert second[0] == 0.0


def __test_bind_any_stateful__():
    """ __bind_any__: state-carrying callee gets a bound partial, state persists """
    acc = _make_stateful()
    parent = _make_state(LAYOUT_PARENT)
    bound = __bind_any__(parent, 2, acc)
    assert parent[2] == (acc, bound)
    assert bound(1.0) == 1.0
    assert bound(2.0) == 3.0


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
    binding from the factory """
    def fake_dispatcher():
        raise AssertionError("the anchor must call the factory's binding")

    def bound_instance(x):
        return x * 10
    fake_dispatcher.__pyne_bind__ = lambda: bound_instance

    parent = _make_state(LAYOUT_PARENT)
    bound = __bind_any__(parent, 2, fake_dispatcher)
    assert bound is bound_instance
    assert parent[2] == (fake_dispatcher, bound_instance)
    assert bound(2) == 20


def __test_bind_any_loop__():
    """ __bind_any_loop__: per-iteration instances, in-place rebind on miss """
    acc = _make_stateful()
    children: list = []
    first = __bind_any_loop__(children, 0, acc)
    second = __bind_any_loop__(children, 1, acc)
    assert [entry[0] for entry in children] == [acc, acc]
    assert first(1.0) == 1.0
    assert first(2.0) == 3.0
    assert second(5.0) == 5.0  # own instance, untouched by first
    # identity miss at an existing index rebinds in place with fresh state
    other = _make_stateful()
    rebound = __bind_any_loop__(children, 0, other)
    assert children[0] == (other, rebound)
    assert rebound(1.0) == 1.0
    assert second(1.0) == 6.0  # the sibling entry keeps its state


def __test_reset__():
    """ reset(): clears child slots of roots, leaves var/series slots alone """
    acc = _make_stateful()
    root = create_root('test·reset', LAYOUT_PARENT)
    try:
        __resolve_slot__(root, 2, acc)
        __grow__(root[3], acc)
        root[0] = 42
        series = root[1]
        reset()
        assert root[2] is None
        assert root[3] == []
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
