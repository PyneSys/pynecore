"""
Behavior tests for the slot-based SeriesTransformer (and its interplay with
the persistent_series / lib_series transformers) plus slot-layout emission.

These run the series-related mini pipeline (PersistentSeries -> LibrarySeries
-> Series -> Persistent -> apply_layout) on inline sources, exec the result
and drive the emitted functions with hand-made state vectors — no import
hook involved. The real ``lib.bar_index`` is advanced between calls because
``SeriesImpl.add`` consults it to convert same-bar adds into ``set``.
"""
import ast
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from pynecore import lib
from pynecore.core.instance_state import _make_state
from pynecore.core.series import SeriesImpl
from pynecore.transformers.lib_series import LibrarySeriesTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.persistent_series import PersistentSeriesTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout
from pynecore.types.na import NA


def _transform(source: str) -> tuple[dict, str]:
    """Run the series mini pipeline on a source string.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = PersistentSeriesTransformer().visit(tree)
    tree = LibrarySeriesTransformer().visit(tree)
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {}
    exec(compile(tree, '<slot-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


@contextmanager
def _bars():
    """Drive ``lib.bar_index`` for the duration of a test; yields a
    ``next_bar()`` callable and restores the original value afterwards."""
    old = lib.bar_index
    lib.bar_index = 0

    def next_bar():
        lib.bar_index += 1

    try:
        yield next_bar
    finally:
        lib.bar_index = old


def __test_simple_series_slot__():
    """ Series declaration becomes an add() on the slot; history works """
    ns, dump = _transform('''
from pynecore import Series

def main(x):
    s: Series[int] = x
    return s, s[1]
''')
    layout = ns['__pyne_slot_layout__']['main']
    assert layout['init'] == (None,)
    assert layout['series'] == ((0, None),)
    assert layout['names'] == ('s',)
    state = _make_state(layout)
    assert isinstance(state[0], SeriesImpl)
    with _bars() as next_bar:
        value, prev = ns['main'](state, 10)
        assert value == 10
        assert isinstance(prev, NA)
        next_bar()
        value, prev = ns['main'](state, 20)
        assert (value, prev) == (20, 10)
    assert '__state__[0].add(x)' in dump
    assert 'Series' not in dump  # import stripped, annotation removed


def __test_series_set_assign__():
    """ Assignment and augmented assignment become set() on the slot """
    ns, dump = _transform('''
from pynecore import Series

def main(x):
    s: Series[float] = x
    s = s * 2.0
    s += 1.0
    return s, s[1]
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with _bars() as next_bar:
        value, prev = ns['main'](state, 3.0)
        assert value == 7.0
        assert isinstance(prev, NA)
        next_bar()
        value, prev = ns['main'](state, 10.0)
        assert (value, prev) == (21.0, 7.0)
    assert '__state__[0].set(s * 2.0)' in dump
    assert '__state__[0].set(s + 1.0)' in dump


def __test_series_parameter__():
    """ A Series parameter loses the wrapper and gets the add() prologue """
    ns, dump = _transform('''
from pynecore import Series

def f(src: Series[float], length):
    return src[1] + length
''')
    layout = ns['__pyne_slot_layout__']['f']
    assert layout['series'] == ((0, None),)
    state = _make_state(layout)
    with _bars() as next_bar:
        assert isinstance(ns['f'](state, 5.0, 100), NA)
        next_bar()
        assert ns['f'](state, 7.0, 100) == 105.0
    assert 'def f(__state__, src: float, length):' in dump
    assert '__state__[0].add(src)' in dump


def __test_nested_series_shadowing__():
    """ A nested def's own Series shadows the parent's in its own scope """
    ns, dump = _transform('''
from pynecore import Series

def main():
    def test(length):
        s: Series[int] = 1
        return s[length]
    s = 1
    s2: Series[float] = 0.5
    return test, s2[0], s
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'main', 'main·test'}
    # main contains a nested def -> scope-qualified state parameter
    assert '__state·main__[0].add(0.5)' in dump
    state = _make_state(layouts['main'])
    with _bars():
        test_fn, s2_now, s_val = ns['main'](state)
        assert (s2_now, s_val) == (0.5, 1)
        assert test_fn.__pyne_layout__ is layouts['main·test']
        inner_state = _make_state(layouts['main·test'])
        assert test_fn(inner_state, 0) == 1


def __test_parent_series_read_from_nested__():
    """ A stateless nested def reads the parent's series history via closure """
    ns, dump = _transform('''
from pynecore import Series

def main(x):
    s: Series[float] = x
    def prev():
        return s[1]
    return prev
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'main'}  # prev carries no state, gets no hidden param
    state = _make_state(layouts['main'])
    with _bars() as next_bar:
        prev = ns['main'](state, 1.0)
        assert isinstance(prev(), NA)
        next_bar()
        prev = ns['main'](state, 2.0)
        assert prev() == 1.0
    assert '__state·main__[0][1]' in dump


def __test_persistent_series_composition__():
    """ PersistentSeries splits into a series slot plus a persistent slot """
    ns, dump = _transform('''
from pynecore import PersistentSeries

def main():
    s: PersistentSeries[float] = 0.5
    s += 1
    return s, s[1]
''')
    layout = ns['__pyne_slot_layout__']['main']
    # slot 0: the series buffer, slot 1: the persistent scalar
    assert layout['series'] == ((0, None),)
    assert layout['init'] == (None, 0.5)
    state = _make_state(layout)
    with _bars() as next_bar:
        value, prev = ns['main'](state)
        assert value == 1.5
        assert isinstance(prev, NA)
        next_bar()
        value, prev = ns['main'](state)
        assert (value, prev) == (2.5, 1.5)
    assert '__state__[1] = __state__[0].add(__state__[1])' in dump
    assert '__state__[1] = __state__[0].set(__state__[1] + 1)' in dump


def __test_lib_series_main_and_nested__():
    """ Library series anchor in main; nested access goes through the
    parent's state parameter (no hardcoded global names anymore) """
    ns, dump = _transform('''
from pynecore import lib

def main():
    a = lib.close[1]
    def nested():
        return lib.high[1]
    return a, nested
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'main'}  # both buffers anchor in main
    layout = layouts['main']
    assert layout['series'] == ((0, None), (1, None))
    state = _make_state(layout)
    with _bars() as next_bar:
        ns['lib'] = SimpleNamespace(close=10.0, high=11.0)
        a, _nested = ns['main'](state)
        assert isinstance(a, NA)
        next_bar()
        ns['lib'] = SimpleNamespace(close=20.0, high=21.0)
        a, nested = ns['main'](state)
        assert a == 10.0
        assert nested() == 11.0
    assert '__lib·close = __state·main__[0].add(lib.close)' in dump
    assert '__state·main__[1][1]' in dump


def __test_max_bars_back__():
    """ Statement-position lib.max_bars_back() becomes an attribute assign """
    ns, dump = _transform('''
from pynecore import lib, Series

def main(x):
    s: Series[float] = x
    lib.max_bars_back(s, 3)
    return s[2]
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    results = []
    with _bars() as next_bar:
        for value in (1.0, 2.0, 3.0, 4.0):
            results.append(ns['main'](state, value))
            next_bar()
    assert isinstance(results[0], NA) and isinstance(results[1], NA)
    assert results[2:] == [1.0, 2.0]
    assert '__state__[0].max_bars_back = 3' in dump
    assert 'lib.max_bars_back' not in dump


def __test_module_level_series_rejected__():
    """ Module-level Series declarations raise a clear transform error """
    with pytest.raises(SyntaxError):
        _transform('''
from pynecore import Series

s: Series[int] = 0
''')
