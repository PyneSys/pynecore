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
from pynecore.transformers.safe_convert_transformer import SafeConvertTransformer
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
    # The counters are Pine ints, floats at runtime: ``range()`` needs the truncation pass
    tree = SafeConvertTransformer().visit(tree)
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
    assert '__resolve_slot·__(__state__, 0, t1)' in dump
    assert '__resolve_slot·__(__state__, 1, t1)' in dump


def __test_fast_path_loop__():
    """ A loop site shares ONE instance across every iteration (TradingView's
    measured law: a var counter in a called function counts straight through
    the iterations of a bar and across bars) """
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
    assert ns['main'](state) == 6    # 1 + 2 + 3: one shared instance
    assert ns['main'](state) == 15   # 4 + 5 + 6: the same instance counts on
    assert type(state[0]) is list and type(state[0][0]) is list  # [state, bar, snap] cell
    assert '__loop_state·__(__state__, 0, t1)' in dump


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
    assert '__resolve_slot·__' not in dump
    assert '__bind_any·__' not in dump


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
    assert '__bind_any·__(__state__, 0, f)' in dump


def __test_uniform_path_redefined_callee_keeps_state__():
    """ A nested callee reached through the anchor is a NEW object every bar
    (``main`` re-runs and re-executes its ``def``), so the identity check
    misses every bar. The rebind must keep the callee's state — its layout is
    unchanged — instead of resetting it, or every stateful Pine method/function
    on the uniform path would silently lose its var/series/history each bar. """
    ns, dump = _transform('''
from pynecore import Persistent

def main():
    def acc():
        p: Persistent[int] = 0
        p += 1
        return p
    f = acc
    return f()
''')
    assert '__bind_any·__(__state·main__, 0, f)' in dump  # uniform, not fast path
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 1
    assert ns['main'](state) == 2  # state survives the per-bar rebind
    assert ns['main'](state) == 3


def __test_uniform_loop__():
    """ An anchored site in a loop shares ONE instance across iterations """
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
    assert ns['main'](state, True) == 3     # 1 + 2: one shared instance
    assert ns['main'](state, True) == 7     # 3 + 4, persisted
    assert ns['main'](state, False) == 203  # rebound to t2: 101 + 102
    assert '__bind_loop·__(__state__, 0, f)' in dump


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
    assert '__resolve_slot·__(__state·main__, 0, t)' in dump


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
    assert '__bind_any·__(__state__, 0, t)' in dump  # uniform, not fast
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
    assert '__resolve_slot·__(__state__, 0, f)' in dump


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
    assert '__bind_any·__' not in dump and '__resolve_slot·__' not in dump


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
        assert '__resolve_slot·__(__state__, 0, fake_pyne_lib_t092.acc)' in dump
        assert 'fake_pyne_lib_t092.helper(x)' in dump  # direct, untouched
        assert '__bind_any·__(__state__, 1, fake_plain_t092.setter)' in dump
    finally:
        del sys.modules['fake_pyne_lib_t092']
        del sys.modules['fake_plain_t092']


def __test_nested_callee_in_attribute_base_gets_own_site__():
    """ A call inside an attribute callee's base becomes a call site of its own """
    ns, dump = _transform('''
from pynecore import Persistent

def main():
    def bump():
        p: Persistent[str] = ''
        p += 'X'
        return p
    return bump().upper()
''')
    layouts = ns['__pyne_slot_layout__']
    assert layouts['main']['children'] == ((0, 'main·bump·0', False),
                                           (1, 'main·<callee>·1', False))
    state = _make_state(layouts['main'])
    assert ns['main'](state) == 'X'
    assert ns['main'](state) == 'XX'  # exactly one bump() per bar
    assert ns['main'](state) == 'XXX'
    assert dump.count('__resolve_slot·__(__state·main__, 0, bump)') == 1


def __test_nested_callee_in_loop_shares_state__():
    """ Loop iterations share the nested callee's instance, and a per-bar
    redefinition of the nested def carries it across bars """
    ns, _ = _transform('''
from pynecore import Persistent

def main():
    def bump():
        p: Persistent[str] = ''
        p += 'X'
        return p
    out = ''
    for _ in range(2):
        out += bump().upper() + '|'
    return out
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 'X|XX|'
    assert ns['main'](state) == 'XXX|XXXX|'
    assert ns['main'](state) == 'XXXXX|XXXXXX|'


def __test_impure_callee_evaluated_once_per_bar__():
    """ The anchored guard and the rebind share one evaluation of the callee """
    ns, _ = _transform('''
def main(log):
    def probe():
        log.append(1)
        return 'x'
    return probe().upper()
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    log: list = []
    for expected in (1, 2, 3, 4):
        assert ns['main'](state, log) == 'X'
        assert len(log) == expected  # two evaluations per bar would double this


def __test_impure_none_callee_in_loop__():
    """ A None callee raises cleanly at the shared anchor, and an alternating
    callee follows the documented uniform-site semantics: the identity miss
    rebinds, so only the state reachable through the PREVIOUS binding's layout
    survives (the last iteration's fresh instance is what carries over) """
    ns, _ = _transform('''
import types
from pynecore import Persistent

def main():
    def use():
        p: Persistent[str] = ''
        p += 'a'
        return p

    def pick(i):
        return types.SimpleNamespace(run=None if i == 1 else use)

    out = []
    for i in range(3):
        try:
            out.append(pick(i).run())
        except TypeError:
            out.append('TE')
    return out
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == ['a', 'TE', 'a']
    assert ns['main'](state) == ['aa', 'TE', 'a']  # i=0 carries i=2's instance on
    assert ns['main'](state) == ['aa', 'TE', 'a']


def __test_stable_impure_callee_binds_the_right_target__():
    """ On an identity hit the anchor must still hold the callee, not a value
    written by a call site nested inside the callee expression """
    ns, _ = _transform('''
import types
from pynecore import Persistent

def leaf():
    p: Persistent[int] = 0
    p += 1
    return p

BOX = types.SimpleNamespace(fn=leaf)
CALLS = []

def make_box():
    CALLS.append('make_box')
    return BOX

def main():
    return make_box().fn()
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 1
    assert ns['main'](state) == 2  # identity hit must call leaf, not make_box
    assert ns['main'](state) == 3
    assert ns['CALLS'] == ['make_box'] * 3  # one base evaluation per bar


def __test_indirect_attribute_callee_still_anchored__():
    """ A path-less callee with no call in its base keeps its plain anchor """
    ns, dump = _transform('''
import types
from pynecore import Persistent

def acc():
    p: Persistent[int] = 0
    p += 1
    return p

HOLDER = [types.SimpleNamespace(run=acc)]

def main():
    return HOLDER[0].run()
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 1
    assert ns['main'](state) == 2
    assert ns['main'](state) == 3
    assert '__bind_any·__(__state__, 0, HOLDER[0].run)' in dump  # no __c·__ binding here
    assert '__c·__' not in dump


def __test_chained_impure_callees__():
    """ Nested impure sites bind independently and each runs once """
    ns, _ = _transform('''
from pynecore import Persistent

def main():
    def bump():
        p: Persistent[int] = 0
        p += 1
        return p
    return str(bump()).zfill(3).lstrip('0')
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    # A Pine int is a double at runtime: the counter prints as a float
    assert ns['main'](state) == '1.0'
    assert ns['main'](state) == '2.0'
    assert ns['main'](state) == '3.0'


def __test_module_level_impure_stateful_callee_rejected__():
    """ Module level rejects a stateful call inside an attribute callee too """
    with pytest.raises(SyntaxError):
        _transform(COUNTER_FUNC + '''
Z = t1().bit_length()
''')


def __test_generated_temporaries_do_not_clobber_user_names__():
    """ A user variable spelled like a generated temporary survives the call """
    ns, dump = _transform('''
class Box:
    def fn(self, v):
        return v

def make_box():
    return Box()

def main():
    __c__ = 'sentinel'
    __b__ = 'anchor'
    __st__ = 'state'
    __i__ = 'index'
    return make_box().fn(__c__), __b__, __st__, __i__
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == ('sentinel', 'anchor', 'state', 'index')
    assert "__c__ = 'sentinel'" in dump  # the user's name is untouched
    assert '__c·__' in dump             # ours carries the middle dot


def __test_runtime_helpers_survive_user_names__():
    """ A user name spelled like a runtime helper neither shadows nor is shadowed """
    ns, dump = _transform(COUNTER_FUNC + '''
__bind_any__ = 'module global'

def main(__slot_state__, __resolve_slot__):
    a = t1()
    b = [x for x in range(t1())]
    return __slot_state__, __resolve_slot__, a, b, __bind_any__
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 'arg1', 'arg2') == ('arg1', 'arg2', 1, [0], 'module global')
    assert "__bind_any__ = 'module global'" in dump  # the user's names are untouched
    assert '__slot_state__ as __slot_state·__' in dump  # ours carry the middle dot


def __test_comprehension_iterable_uniform_site__():
    """ A comprehension iterable holds no walrus — the whole guard is a helper """
    ns, dump = _transform('''
def main(g):
    return [x for x in g()]
''')
    assert '__bind_slot·__(__state__, 0, g)' in dump
    assert ':=' not in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    seen = []

    def g():
        seen.append(1)
        return [1, 2]

    assert ns['main'](state, g) == [1, 2]
    assert ns['main'](state, g) == [1, 2]
    assert len(seen) == 2  # callee evaluated once per bar


def __test_comprehension_iterable_fast_site__():
    """ A state-carrying callee in a comprehension iterable keeps its instance """
    ns, dump = _transform(COUNTER_FUNC + '''
def main():
    return [x for x in range(t1())]
''')
    layouts = ns['__pyne_slot_layout__']
    # The OUTERMOST iterable runs once, in the enclosing scope: not a loop site
    assert layouts['main']['children'] == ((0, 'main·t1·0', False),)
    assert '__slot_state·__(__state__, 0, t1)' in dump
    state = _make_state(layouts['main'])
    assert ns['main'](state) == [0]
    assert ns['main'](state) == [0, 1]


def __test_comprehension_iterable_pathless_callee__():
    """ ``f().g()`` in a comprehension iterable compiles and runs once per bar """
    ns, dump = _transform('''
CALLS = []

class Box:
    def items(self):
        return [1, 2]

def make_box():
    CALLS.append('make_box')
    return Box()

def main():
    return [x for x in make_box().items()]
''')
    assert ':=' not in dump.split('def main')[1]
    # The nested state-carrying base gets its own (also walrus-free) site
    assert '__bind_slot·__(__state__, 1, make_box(__slot_state·__(__state__, 0, ' \
           'make_box)).items)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == [1, 2]
    assert ns['main'](state) == [1, 2]
    assert ns['CALLS'] == ['make_box'] * 2  # callee expression evaluated once per bar


def __test_comprehension_iterable_in_loop_shares_state__():
    """ A loop-shaped iterable site uses the shared helper form (walrus-free
    by construction, so it is legal in the iterable position) """
    ns, dump = _transform(COUNTER_FUNC + '''
def main():
    out = []
    for _ in range(3):
        out.append([x for x in range(t1())])
    return out
''')
    layouts = ns['__pyne_slot_layout__']
    assert layouts['main']['children'][0] == (0, 'main·t1·0', True)
    assert '__loop_state·__(__state__, 0, t1)' in dump
    state = _make_state(layouts['main'])
    assert ns['main'](state) == [[0], [0, 1], [0, 1, 2]]      # 1, 2, 3 shared
    assert ns['main'](state) == [list(range(n)) for n in (4, 5, 6)]


def __test_nested_comprehension_iterable_is_a_loop_site__():
    """ A LATER generator's iterable runs per element -> shared loop site """
    ns, dump = _transform('''
from pynecore import Persistent

def bump(_a):
    p: Persistent[int] = 0
    p += 1
    return range(p)

def main(items):
    return [y for a in items for y in bump(a)]
''')
    layouts = ns['__pyne_slot_layout__']
    assert layouts['main']['children'][0] == (0, 'main·bump·0', True)
    assert '__loop_state·__(__state__, 0, bump)' in dump
    assert ':=' not in dump
    state = _make_state(layouts['main'])
    assert ns['main'](state, [10, 20]) == [0, 0, 1]           # p = 1 then 2
    assert ns['main'](state, [10, 20]) == [0, 1, 2, 0, 1, 2, 3]


def __test_comprehension_element_sites_are_loop_sites__():
    """ Element and condition sites run per element — the shared loop helper
    serves them (the iterable's straight-line site keeps its own helper) """
    _, dump = _transform('''
def main(g, h):
    return [h(x) for x in g() if h(x)]
''')
    assert '__bind_slot·__(__state__, 0, g)' in dump
    assert '__bind_loop·__' in dump  # element/condition sites share one anchor
