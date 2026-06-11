"""
Behavior tests for the slot-based PersistentTransformer + slot-layout emission.

These run a minimal pipeline (PersistentTransformer + apply_layout) on inline
sources, exec the result and drive the emitted functions with hand-made state
vectors — no import hook involved.
"""
import ast

import pytest

from pynecore.core.instance_state import _make_state
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout


def _transform(source: str) -> tuple[dict, str]:
    """Run the minimal slot pipeline on a source string.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = PersistentTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {}
    exec(compile(tree, '<slot-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


def __test_literal_counter__():
    """ Literal init moves into the layout; += 1 stays a plain slot AugAssign """
    ns, dump = _transform('''
from pynecore import Persistent

def main():
    p: Persistent[int] = 0
    p += 1
    return p
''')
    layout = ns['__pyne_slot_layout__']['main']
    assert layout['init'] == (0,)
    assert layout['names'] == ('p',)
    assert ns['main'].__pyne_layout__ is layout
    state = _make_state(layout)
    assert ns['main'](state) == 1
    assert ns['main'](state) == 2
    assert state == [2]
    assert '__state__[0] += 1' in dump
    assert 'Persistent' not in dump  # import stripped, annotation removed


def __test_lazy_init__():
    """ Non-literal init keeps the flag-guarded lazy pattern """
    ns, _ = _transform('''
from pynecore import Persistent

def main(length):
    q: Persistent[float] = float(length)
    return q
''')
    layout = ns['__pyne_slot_layout__']['main']
    assert layout['init'] == (None, False)
    assert layout['names'] == ('q', 'q·flag')
    state = _make_state(layout)
    assert ns['main'](state, 5) == 5.0
    assert ns['main'](state, 7) == 5.0  # initializer must not run again
    assert state == [5.0, True]


def __test_kahan__():
    """ += with non-literal value emits the exact Kahan sequence """
    ns, dump = _transform('''
from pynecore import Persistent

def main(x):
    p: Persistent[float] = 0.0
    p += x
    return p
''')
    layout = ns['__pyne_slot_layout__']['main']
    assert layout['names'] == ('p', 'p·kahan')
    assert layout['init'] == (0.0, 0.0)
    state = _make_state(layout)

    # reference implementation of the same algorithm
    ref_sum = ref_comp = 0.0
    for value in [0.1] * 10 + [1e16, 1.0, -1e16]:
        corrected = value - ref_comp
        new_sum = ref_sum + corrected
        ref_comp = (new_sum - ref_sum) - corrected
        ref_sum = new_sum
        assert ns['main'](state, value) == ref_sum
    assert state == [ref_sum, ref_comp]
    assert '__kahan_corrected__' in dump


def __test_varip_layout__():
    """ IBPersistent slots are recorded in the layout's varip tuple """
    ns, _ = _transform('''
from pynecore import Persistent
from pynecore.types import IBPersistent

def main():
    v: IBPersistent[int] = 0
    p: Persistent[int] = 10
    v += 1
    p += 1
    return v + p
''')
    layout = ns['__pyne_slot_layout__']['main']
    assert layout['init'] == (0, 10)
    assert layout['varip'] == (0,)
    state = _make_state(layout)
    assert ns['main'](state) == 12


def __test_nested_shadowing__():
    """ A nested def's own Persistent shadows the parent's in its own scope """
    ns, dump = _transform('''
from pynecore import Persistent

def main():
    p: Persistent[float] = 1
    p += 2.5
    def test():
        p: Persistent[float] = 1
        p += 1
        return p
    return test
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'main', 'main·test'}
    # main contains a nested def -> scope-qualified state parameter
    assert '__state·main__[0] += 2.5' in dump
    state = _make_state(layouts['main'])
    inner = ns['main'](state)
    assert inner.__pyne_layout__ is layouts['main·test']
    inner_state = _make_state(layouts['main·test'])
    assert inner(inner_state) == 2
    assert inner(inner_state) == 3
    assert state == [3.5]  # parent's own p is untouched by the shadow


def __test_parent_access_from_nested__():
    """ A stateless nested def reads/uses the parent's persistent via closure """
    ns, _ = _transform('''
from pynecore import Persistent

def main():
    p: Persistent[int] = 0
    def bump():
        return p + 1
    p = bump()
    p = bump()
    return p
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'main'}  # bump carries no state, gets no hidden param
    state = _make_state(layouts['main'])
    assert ns['main'](state) == 2
    assert ns['main'](state) == 4
    assert state == [4]


def __test_walrus_write__():
    """ Walrus write to a persistent stays an expression and yields the value """
    ns, dump = _transform('''
from pynecore import Persistent

def main():
    p: Persistent[int] = 0
    y = ((p := p + 1) * 2)
    return y
''')
    layout = ns['__pyne_slot_layout__']['main']
    state = _make_state(layout)
    assert ns['main'](state) == 2
    assert ns['main'](state) == 4
    assert state == [2]
    assert '__setitem__' in dump


def __test_persistent_as_call_argument__():
    """ Persistent references are rewritten inside call arguments """
    ns, _ = _transform('''
from pynecore import Persistent

def helper(a):
    return a * 2

def main():
    p: Persistent[int] = 21
    return helper(p)
''')
    assert set(ns['__pyne_slot_layout__']) == {'main'}
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 42


def __test_local_shadows_parent_persistent__():
    """ A plain local assignment shadows a same-named parent persistent """
    ns, _ = _transform('''
from pynecore import Persistent

def main():
    p: Persistent[int] = 7
    def inner():
        p = 100
        return p
    inner()
    return p
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 7  # inner's local write must not touch the slot
    assert state == [7]


def __test_annotated_local_shadows_parent_persistent__():
    """ An annotated local declaration shadows a same-named parent persistent """
    ns, _ = _transform('''
from pynecore import Persistent

def main():
    p: Persistent[int] = 7
    def inner():
        p: int = 100
        return p
    inner()
    return p
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 7  # inner's annotated local must not touch the slot
    assert state == [7]


def __test_module_level_persistent_rejected__():
    """ Module-level Persistent declarations raise a clear transform error """
    with pytest.raises(SyntaxError):
        _transform('''
from pynecore import Persistent

p: Persistent[int] = 0
''')
