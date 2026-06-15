"""
Behavior tests for the slot-based FunctionIsolationTransformer.

These run the slot mini pipeline (Series -> Persistent -> FunctionIsolation
-> apply_layout) on inline sources, exec the result and drive the emitted
functions with hand-made state vectors. The emitted module imports the real
``pynecore.core.instance_state`` helpers, so the full call-site machinery
(fast path, child lists, anchored binds) is exercised end to end.
"""
import ast
import sys
import types

import pytest

from pynecore.core.instance_state import _make_state
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout


def _transform(source: str) -> tuple[dict, str]:
    """Run the slot mini pipeline on a source string.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {}
    exec(compile(tree, '<slot-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


COUNTER_FUNC = '''
from pynecore import Persistent

def t1():
    p: Persistent[int] = 0
    p += 1
    return p
'''


def __test_fast_path_straight_line__():
    """ Two sites to the same stateful callee get two independent child slots """
    ns, dump = _transform(COUNTER_FUNC + '''
def main():
    a = t1()
    b = t1()
    return a, b
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'t1', 'main'}
    assert layouts['main']['children'] == ((0, 'main·t1·0', False), (1, 'main·t1·1', False))
    state = _make_state(layouts['main'])
    assert ns['main'](state) == (1, 1)
    assert ns['main'](state) == (2, 2)  # both instances persist independently
    assert '__resolve_slot__(__state__, 0, t1)' in dump
    assert '__resolve_slot__(__state__, 1, t1)' in dump


def __test_fast_path_loop__():
    """ A loop site keeps one instance per iteration via the child list """
    ns, dump = _transform(COUNTER_FUNC + '''
def main():
    total = 0
    for _ in range(3):
        total += t1()
    return total
''')
    layouts = ns['__pyne_slot_layout__']
    assert layouts['main']['children'] == ((0, 'main·t1·0', True),)
    state = _make_state(layouts['main'])
    assert ns['main'](state) == 3   # three fresh instances
    assert ns['main'](state) == 6   # the same three instances again
    assert len(state[0]) == 3
    assert '__cnt_0__ = 0' in dump
    assert '__chl_0__ = __state__[0]' in dump
    assert '__grow__(__chl_0__, t1)' in dump


def __test_fast_path_loop_len_shadow__():
    """ A script variable named ``len`` must not break the loop counter guard """
    ns, dump = _transform(COUNTER_FUNC + '''
def main():
    len = 7  # shadows the builtin, like a common Pine input name
    total = 0
    for _ in range(3):
        total += t1()
    return total + len
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 10   # three fresh instances (1+1+1) + len(7)
    assert ns['main'](state) == 13   # same three instances (2+2+2 -> 6) + 7
    assert 'len(__chl_0__)' not in dump   # never a shadowable builtin call
    assert '__chl_0__.__len__()' in dump


def __test_direct_path_stateless__():
    """ Provably stateless callees stay plain calls, nobody grows a layout """
    ns, dump = _transform('''
def helper(x):
    return x * 2

def main(x):
    return helper(x)
''')
    assert ns['__pyne_slot_layout__'] == {}
    assert ns['main'](5) == 10  # no hidden parameters anywhere
    assert '__resolve_slot__' not in dump
    assert '__bind_any__' not in dump


def __test_carrier_fixpoint__():
    """ State-carrying propagates through the call graph (main -> t -> u) """
    ns, _ = _transform('''
from pynecore import Persistent

def u():
    p: Persistent[int] = 0
    p += 1
    return p

def t():
    return u() * 10

def main():
    return t()
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'u', 't', 'main'}  # t and main carry via child slots
    state = _make_state(layouts['main'])
    assert ns['main'](state) == 10
    assert ns['main'](state) == 20  # u's instance persists through the chain


def __test_uniform_path__():
    """ A function-valued callee is anchored; swap rebinds with fresh state """
    ns, dump = _transform(COUNTER_FUNC + '''
def t2():
    p: Persistent[int] = 100
    p += 1
    return p

def main(flag):
    f = t1 if flag else t2
    return f()
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, True) == 1
    assert ns['main'](state, True) == 2     # identity hit, state persists
    assert ns['main'](state, False) == 101  # rebind: fresh t2 instance
    assert ns['main'](state, True) == 1     # swap back: fresh again (documented)
    assert '__bind_any__(__state__, 0, f)' in dump


def __test_uniform_loop__():
    """ An anchored site in a loop keeps one instance per iteration """
    ns, dump = _transform(COUNTER_FUNC + '''
def t2():
    p: Persistent[int] = 100
    p += 1
    return p

def main(flag):
    f = t1 if flag else t2
    total = 0
    for _ in range(2):
        total += f()
    return total
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, True) == 2     # 1 + 1, two fresh instances
    assert ns['main'](state, True) == 4     # 2 + 2, both persisted
    assert ns['main'](state, False) == 202  # both iterations rebound to t2
    assert '__bind_any_loop__(__chl_0__, __i__, f)' in dump


def __test_uniform_loop_len_shadow__():
    """ The anchored loop guard is shadow-proof against a ``len`` variable too """
    ns, dump = _transform(COUNTER_FUNC + '''
def t2():
    p: Persistent[int] = 100
    p += 1
    return p

def main(flag):
    len = 5  # shadows the builtin
    f = t1 if flag else t2
    total = 0
    for _ in range(2):
        total += f()
    return total + len
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, True) == 7      # 1 + 1 + len(5)
    assert ns['main'](state, True) == 9      # 2 + 2 + 5
    assert 'len(__chl_0__)' not in dump
    assert '__chl_0__.__len__()' in dump


def __test_nested_def_fast_path__():
    """ A nested def's call sites use the fast path: the function object is
    recreated every invocation but the per-site state persists """
    ns, dump = _transform('''
from pynecore import Persistent

def main():
    def t():
        p: Persistent[int] = 0
        p += 1
        return p
    a = t()
    b = t()
    return a, b
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'main', 'main·t'}
    state = _make_state(layouts['main'])
    assert ns['main'](state) == (1, 1)
    assert ns['main'](state) == (2, 2)
    assert '__resolve_slot__(__state·main__, 0, t)' in dump


def __test_decorated_def_uniform__():
    """ A decorated def routes uniform (the name's runtime value is the
    decorator's result) and gets its layout through the attach decorator """
    ns, dump = _transform('''
from pynecore import Persistent

def deco(func):
    return func

@deco
def t():
    p: Persistent[int] = 0
    p += 1
    return p

def main():
    return t()
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'t', 'main'}
    state = _make_state(layouts['main'])
    assert ns['main'](state) == 1
    assert ns['main'](state) == 2  # anchored instance persists
    assert '__bind_any__(__state__, 0, t)' in dump  # uniform, not fast
    # the attach decorator sits innermost, tagging the raw function
    assert "@__attach_layout__(__pyne_slot_layout__['t'])" in dump
    assert 't.__pyne_layout__' not in dump  # no post-def attach for decorated defs


def __test_duplicate_def_names_get_own_scopes__():
    """ Repeated definitions of one name keep separate layouts; call sites
    resolve to the last definition, like the runtime name binding """
    ns, dump = _transform('''
from pynecore import Persistent

def f():
    p: Persistent[int] = 0
    p += 1
    return p

def f():
    q: Persistent[int] = 100
    q += 1
    return q

def main():
    return f()
''')
    layouts = ns['__pyne_slot_layout__']
    assert set(layouts) == {'f', 'f·2', 'main'}
    assert layouts['f']['names'] == ('p',)
    assert layouts['f·2']['names'] == ('q',)
    state = _make_state(layouts['main'])
    assert ns['main'](state) == 101  # the second definition wins
    assert ns['main'](state) == 102
    assert '__resolve_slot__(__state__, 0, f)' in dump


def __test_builtins_skipped__():
    """ Builtin calls are not isolation territory """
    ns, dump = _transform('''
def main(xs):
    n = len(xs)
    print(n, end='')
    return n
''')
    assert ns['__pyne_slot_layout__'] == {}
    assert ns['main']([1, 2, 3]) == 3
    assert '__bind_any__' not in dump and '__resolve_slot__' not in dump


def __test_module_level_stateful_call_rejected__():
    """ A module-level call to a stateful function raises a transform error """
    with pytest.raises(SyntaxError):
        _transform(COUNTER_FUNC + '''
x = t1()
''')


def __test_test_function_exemption__():
    """ __test_*__ functions stay raw: no hidden parameter, no slots """
    ns, dump = _transform(COUNTER_FUNC + '''
def main():
    return t1()

def __test_foo__(file_reader):
    return file_reader()
''')
    assert '__test_foo__' not in ns['__pyne_slot_layout__']
    assert ns['__test_foo__'](lambda: 42) == 42
    assert 'file_reader()' in dump


def __test_cross_module_classification__():
    """ Imported callees are classified at transform time: layout attribute
    -> fast path, transformed-module marker -> direct, unknown -> uniform """
    transformed_mod = types.ModuleType('fake_pyne_lib_t092')
    exec('''
__pyne_slot_layout__ = {'acc': {'init': (0.0,), 'series': (), 'varip': (),
                                'children': (), 'names': ('total',)}}

def acc(__state__, x):
    __state__[0] += x
    return __state__[0]

acc.__pyne_layout__ = __pyne_slot_layout__['acc']

def helper(x):
    return x + 1
''', transformed_mod.__dict__)
    plain_mod = types.ModuleType('fake_plain_t092')
    exec('''
def setter(x):
    return -x
''', plain_mod.__dict__)
    sys.modules['fake_pyne_lib_t092'] = transformed_mod
    sys.modules['fake_plain_t092'] = plain_mod
    try:
        ns, dump = _transform('''
import fake_pyne_lib_t092
import fake_plain_t092

def main(x):
    return fake_pyne_lib_t092.acc(x), fake_pyne_lib_t092.helper(x), fake_plain_t092.setter(x)
''')
        state = _make_state(ns['__pyne_slot_layout__']['main'])
        assert ns['main'](state, 1.0) == (1.0, 2.0, -1.0)
        assert ns['main'](state, 2.0) == (3.0, 3.0, -2.0)  # acc state persisted
        assert '__resolve_slot__(__state__, 0, fake_pyne_lib_t092.acc)' in dump
        assert 'fake_pyne_lib_t092.helper(x)' in dump  # direct, untouched
        assert '__bind_any__(__state__, 1, fake_plain_t092.setter)' in dump
    finally:
        del sys.modules['fake_pyne_lib_t092']
        del sys.modules['fake_plain_t092']
