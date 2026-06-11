"""
Behavior tests for overload dispatch on the slot scheme.

The sources use the real ``pynecore.core.overload`` decorator and run through
the slot mini pipeline (Series -> Persistent -> FunctionIsolation ->
apply_layout). Overloaded names are decorated defs, so call sites take the
uniform route: the anchor stores an anchored dispatcher from
``__pyne_bind__``, which keeps one bound instance PER IMPLEMENTATION — the
per-anchor, per-implementation state semantics the plan's "overload anchor"
design specifies.

The overload registry is process-global and keyed by the function's
``__module__``-qualified name, so every test execs its module under a unique
``__name__``.
"""
import ast

import pytest

from pynecore.core.instance_state import _make_state
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout


def _transform(source: str, mod_name: str) -> tuple[dict, str]:
    """Run the slot mini pipeline and exec the result under a module name.

    :param source: Pyne-style module source.
    :param mod_name: Unique module name (isolates the overload registry).
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {'__name__': mod_name}
    exec(compile(tree, '<slot-overload-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


ACC_SRC = '''
from pynecore import Persistent
from pynecore.core.overload import overload

@overload
def acc(x: int):
    total: Persistent[int] = 0
    total += x
    return total

@overload
def acc(x: str):
    joined: Persistent[str] = ''
    joined += x
    return joined
'''


def __test_overload_per_impl_state__():
    """ One call site, alternating implementations: each keeps its own
    persistent instance in the same anchor """
    ns, dump = _transform(ACC_SRC + '''
def main(v):
    return acc(v)
''', 'ovl_mod_a')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'acc', 'acc·2', 'main'}
    assert layouts['acc']['names'][0] == 'total'
    assert layouts['acc·2']['names'][0] == 'joined'
    state = _make_state(layouts['main'])
    assert ns['main'](state, 1) == 1
    assert ns['main'](state, 'a') == 'a'
    assert ns['main'](state, 2) == 3      # int impl state survived the str call
    assert ns['main'](state, 'b') == 'ab'  # and the str impl state too
    # decorated def -> uniform route, never the fast path
    assert '__bind_any__(__state__, 0, acc)' in dump
    assert '__resolve_slot__' not in dump
    # both implementations got their layout through the attach decorator
    assert "@__attach_layout__(__pyne_slot_layout__['acc'])" in dump
    assert "@__attach_layout__(__pyne_slot_layout__['acc·2'])" in dump
    # the @wraps __dict__ copy must not leave a layout on the dispatcher
    assert '__pyne_layout__' not in ns['acc'].__dict__
    assert hasattr(ns['acc'], '__pyne_bind__')


def __test_overload_independent_anchors__():
    """ Two call sites of the same dispatcher hold independent instances """
    ns, _ = _transform(ACC_SRC + '''
def main(v):
    return acc(v), acc(v)
''', 'ovl_mod_b')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == (1, 1)
    assert ns['main'](state, 2) == (3, 3)


def __test_overload_stateless_impl_raw__():
    """ A stateless implementation has no hidden parameter and is called raw,
    next to a state-carrying sibling """
    ns, _ = _transform('''
from pynecore import Persistent
from pynecore.core.overload import overload

@overload
def f(x: int):
    p: Persistent[int] = 0
    p += x
    return p

@overload
def f(x: str):
    return x + '!'

def main(v):
    return f(v)
''', 'ovl_mod_c')
    layouts = ns['__pyne_slot_layout__']
    assert 'f' in layouts and 'f·2' not in layouts  # the str impl has no slots
    state = _make_state(layouts['main'])
    assert ns['main'](state, 'ab') == 'ab!'
    assert ns['main'](state, 2) == 2
    assert ns['main'](state, 3) == 5


def __test_overload_kwargs_path__():
    """ Keyword calls go through the signature-bind path with the hidden
    parameter excluded """
    ns, _ = _transform(ACC_SRC + '''
def main(v):
    return acc(x=v)
''', 'ovl_mod_d')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 5) == 5
    assert ns['main'](state, 7) == 12


def __test_overload_no_match__():
    """ No matching implementation raises TypeError """
    ns, _ = _transform(ACC_SRC + '''
def main(v):
    return acc(v)
''', 'ovl_mod_e')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with pytest.raises(TypeError, match='No matching implementation'):
        ns['main'](state, 1.5)


def __test_overload_direct_call_fallback__():
    """ Calling the dispatcher without an anchor uses its own shared bound
    cache: one module-lifetime instance per implementation """
    ns, _ = _transform(ACC_SRC, 'ovl_mod_f')
    acc = ns['acc']
    assert acc(1) == 1
    assert acc('x') == 'x'
    assert acc(2) == 3  # shared instance persists across direct calls


def __test_overload_reexec_keeps_dispatcher__():
    """ Re-executing a module re-decorates the same lines: the dispatcher
    survives and rebinds to the fresh implementation functions """
    src = ACC_SRC + '''
def main(v):
    return acc(v)
'''
    ns1, _ = _transform(src, 'ovl_mod_g')
    state1 = _make_state(ns1['__pyne_slot_layout__']['main'])
    assert ns1['main'](state1, 1) == 1
    assert ns1['main'](state1, 2) == 3

    ns2, _ = _transform(src, 'ovl_mod_g')
    assert ns2['acc'] is ns1['acc']  # same dispatcher object
    state2 = _make_state(ns2['__pyne_slot_layout__']['main'])
    assert ns2['main'](state2, 4) == 4  # fresh state, rebound implementation
    # the old anchor detects the swapped function and rebinds with fresh state
    assert ns1['main'](state1, 1) == 1
