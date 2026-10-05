"""Exported bindings keep their state visible across bool-mode boundaries."""
import pytest

from pynecore import lib
from pynecore.core import instance_state
from pynecore.core.overload import Implementation, _anchored
from pynecore.core.pine_export import Exported
from pynecore.core.series import inline_series
from pynecore.types import na as na_types


@pytest.fixture(params=[(False, False), (False, True), (True, False), (True, True)])
def modes(request, monkeypatch):
    """Exercise both modes after a three-state module has been loaded."""
    caller, library = request.param
    monkeypatch.setattr(na_types, '_bool_na_seen', True)
    monkeypatch.setattr(na_types, '_bool_na', caller)
    monkeypatch.setattr(lib, 'bar_index', 0)
    return caller, library


@pytest.fixture
def root():
    key = 'test_156_export_state'
    layout = {'init': (None, None), 'series': (), 'varip': (),
              'children': ((0, 'export', False), (1, 'loop_export', True))}
    state = instance_state.create_root(key, layout)
    try:
        yield key, state
    finally:
        instance_state.discard_root(key)


def _accumulator(library_mode, layout=None):
    def accumulate(__state__, value: float):
        assert na_types._bool_na is library_mode
        __state__[0] += value
        return __state__[0]
    accumulate.__pyne_layout__ = layout if layout is not None else {
        'init': (0.0,), 'series': (), 'varip': (), 'children': ()}
    return accumulate


def _export(target, library_mode):
    exported = Exported()
    exported.set(target, library_mode)
    return exported


def __test_exported_state_survives_discarded_execution__(modes, root):
    """A developing security re-run keeps committed state and undoes its trial."""
    caller, library = modes
    key, state = root
    exported = _export(_accumulator(library), library)
    bound = instance_state.__bind_any__(state, 0, exported)
    assert bound(5.0) == 5.0
    snapshot = instance_state.RootChildSnapshot([key])
    snapshot.save()
    for _ in range(2):
        assert bound(99.0) == 104.0
        snapshot.restore()
        assert state[0] == (exported, bound)
    assert instance_state.__bind_slot__(state, 0, exported)(2.0) == 7.0
    assert na_types._bool_na is caller


def __test_exported_overloads_restore_each_machine__(modes, root):
    """The bool wrapper exposes a dispatcher's per-implementation machines."""
    caller, library = modes
    key, state = root
    numeric = Implementation(_accumulator(library))

    def text(__state__, value: str):
        assert na_types._bool_na is library
        __state__[0] += len(value)
        return __state__[0]
    text.__pyne_layout__ = numeric.func.__pyne_layout__
    textual = Implementation(text)

    def target():
        raise AssertionError('the call must use the anchored dispatcher')

    target.__pyne_bind__ = lambda pin=None: _anchored([numeric, textual], 'export', pin=pin)
    exported = _export(target, library)
    bound = instance_state.__bind_any__(state, 0, exported)
    assert bound(5.0) == 5.0
    snapshot = instance_state.RootChildSnapshot([key])
    snapshot.save()
    assert bound(99.0) == 104.0
    assert bound('discarded') == 9.0
    snapshot.restore()
    assert state[0] == (exported, bound)
    assert bound(2.0) == 7.0
    assert bound('ok') == 2.0
    assert na_types._bool_na is caller


def __test_exported_inline_history_survives_rollback__(modes, root):
    """A closure-held series is restored rather than rebound to an empty buffer."""
    caller, library = modes
    key, state = root
    exported = _export(inline_series, library)
    bound = instance_state.__bind_any__(state, 0, exported)
    assert bound(5.0, 0) == 5.0
    lib.bar_index = 1
    snapshot = instance_state.RootChildSnapshot([key])
    snapshot.save()
    assert bound(99.0, 1) == 5.0
    snapshot.restore()
    assert state[0] == (exported, bound)
    assert bound(2.0, 1) == 5.0
    assert na_types._bool_na is caller


def __test_exported_redefinition_carries_the_same_state__(modes, root):
    """A fresh function with the same layout retains its anchored state vector."""
    caller, library = modes
    _key, state = root
    original = _accumulator(library)
    assert instance_state.__bind_any__(state, 0, _export(original, library))(5.0) == 5.0
    redefined = _accumulator(library, original.__pyne_layout__)
    assert instance_state.__bind_any__(state, 0, _export(redefined, library))(2.0) == 7.0
    different = _accumulator(library)
    assert instance_state.__bind_any__(state, 0, _export(different, library))(2.0) == 2.0
    assert na_types._bool_na is caller


def __test_exported_loop_machine_keeps_its_bar_start_baseline__(modes, root):
    """Builtin machines behind an export rederive each loop call from bar start."""
    caller, library = modes
    _key, state = root
    layout = {'init': (0.0,), 'series': (), 'varip': (), 'children': (), 'compacted': True}
    exported = _export(_accumulator(library, layout), library)
    assert instance_state.__bind_loop__(state, 1, exported)(1.0) == 1.0
    assert instance_state.__bind_loop__(state, 1, exported)(2.0) == 2.0
    lib.bar_index = 1
    assert instance_state.__bind_loop__(state, 1, exported)(3.0) == 5.0
    assert na_types._bool_na is caller
