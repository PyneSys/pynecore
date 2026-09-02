"""
The per-instance pin: one shared body, one overload answer per INSTANCE.

A generic helper has ONE body and as many instantiations as it has call-site
contexts. An overload site inside it can therefore justify ``'i'`` under an int
caller and nothing at all under a float one -- both true, so the site cannot
carry a constant pin and used to fall back to dispatching on the values. That
is wrong for the int context, which is the whole point of Pine's static
``int``: ``f(1)`` runs ``g(0.5)``, and TradingView still picks the INT
implementation there.

The channel closes it. The type pass marks the sites a body resolves per
instance, the isolation pass reserves ONE state slot for the vector of those
answers, and every call site hands the callee the vector belonging to the
instance it creates. The site then reads its own entry out of the slot instead
of a constant. ``PYNE_NO_TYPE_PIN=1`` switches the whole thing off, because the
characters still reach the ordinary selector through the same binder.
"""
import ast

import pytest

from pynecore.core.instance_state import (
    RootVarSnapshot, __bind_any__, _make_state, _restore_collected, create_root, discard_root,
)
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import get_varying, get_vector
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout

#: An overload group whose two implementations are told apart by their result:
#: 1.0 is the int one, 2.0 the float one.
GROUP = '''
from pynecore.core.overload import overload

@overload
def g(x: int) -> float:
    return 1.0

@overload
def g(x: float) -> float:
    return 2.0
'''

#: The shape the channel exists for: ``x / 2`` is int-TYPED under an int
#: caller with a fractional VALUE, so the values alone would answer 2.0 there.
SHARED_BODY = '''
def f(x):
    return g(x / 2)
'''


def _transform(source: str, mod_name: str) -> tuple[dict, str]:
    """Run the slot mini pipeline WITH the type pass and exec the result.

    The type pass sits exactly where the real pipeline puts it: after the
    closure arguments, before the series and isolation passes.

    :param source: Pyne-style module source
    :param mod_name: Unique module name (isolates the overload registry)
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = PineTypeTransformer(None).visit(tree)
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {'__name__': mod_name}
    exec(compile(tree, '<instance-pin-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


def _defs(source: str) -> dict[str, ast.FunctionDef]:
    """Infer a snippet and return its definitions by name (the LAST one wins)."""
    tree = ast.parse(source)
    infer_module(tree, 'test')
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)}


# --- what the type pass marks --------------------------------------------


def __test_a_disagreeing_site_is_marked_instance_varying__():
    """The body that answers two ways carries the site; the callers do not"""
    defs = _defs(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''')
    varying = get_varying(defs['f'])
    assert varying is not None
    assert [ast.unparse(site) for site in varying] == ['g(x / 2)']
    assert get_varying(defs['main']) is None


def __test_each_call_site_carries_its_own_instance_vector__():
    """The int caller configures ``'i'``; the float caller configures nothing"""
    tree = ast.parse(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''')
    infer_module(tree, 'test')
    vectors = [get_vector(node) for node in ast.walk(tree)
               if isinstance(node, ast.Call) and ast.unparse(node.func) == 'f']
    # The all-None vector IS the layout default, so it is never stamped: an
    # unconfigured instance already dispatches from the values
    assert vectors == [('i',), None]


def __test_two_sites_of_one_closure_get_a_vector_each__():
    """
    A closure called before and after its capture widens is TWO instances.

    ``f`` takes no arguments, so nothing but the type of the captured ``x``
    tells its two call sites apart. Superseding on everything BUT the call
    node collapsed them into one context: the site inside ``f`` then had one
    answer for both, no varying mark, and the int call dispatched on the
    fractional value 0.5 into the float implementation.
    """
    source = GROUP + '''
def main(r: int):
    x = 1 / 2
    def f():
        return g(x)
    first = f()
    x = x + 0.5
    second = f()
    return first, second
'''
    defs = _defs(source)
    assert [ast.unparse(site) for site in get_varying(defs['f']) or ()] == ['g(x)']

    tree = ast.parse(source)
    infer_module(tree, 'test')
    vectors = [get_vector(node) for node in ast.walk(tree)
               if isinstance(node, ast.Call) and ast.unparse(node.func) == 'f']
    assert vectors == [('i',), None]

    ns, _ = _transform(source, 'pin_mod_two_sites')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (1.0, 2.0)


# --- emission -------------------------------------------------------------


def __test_the_varying_site_reads_its_pin_out_of_the_slot__():
    """The callee gets a pin slot, its site addresses it, the callers fill it"""
    _, dump = _transform(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_emit')
    assert "'pin': 0" in dump
    assert '__bind_any·__(__state__, 1, g, __state__[0][0])' in dump
    assert "__resolve_slot·__(__state__, 0, f, ('i',))" in dump
    assert '__resolve_slot·__(__state__, 1, f)' in dump


def __test_a_body_that_agrees_gets_no_slot_and_no_extra_argument__():
    """Nothing varies, so the emitted shape is the one test_092 pins down"""
    _, dump = _transform('''
from pynecore import Persistent

def acc(x):
    total: Persistent[int] = 0
    total += x
    return total

def main(r: int):
    return acc(r), acc(r * 1.5)
''', 'ipin_mod_plain')
    assert "'pin'" not in dump
    assert '__resolve_slot·__(__state__, 0, acc)' in dump
    assert '__resolve_slot·__(__state__, 1, acc)' in dump


# --- runtime --------------------------------------------------------------


def __test_one_body_dispatches_two_ways_per_instance__():
    """``f(1)`` runs ``g(0.5)`` and still lands on the INT implementation"""
    ns, _ = _transform(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_run')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == (1.0, 2.0)


def __test_the_channel_switches_off_with_the_pin__(monkeypatch):
    """``PYNE_NO_TYPE_PIN=1`` puts both instances back on value dispatch"""
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    ns, _ = _transform(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_off')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == (2.0, 2.0)


def __test_the_loop_site_configures_its_shared_instance__():
    """A loop site keeps ONE instance, so it is configured once, at creation"""
    ns, _ = _transform(GROUP + SHARED_BODY + '''
def main(r: int):
    out = []
    for _ in range(2):
        out.append(f(r))
    for _ in range(2):
        out.append(f(r * 1.5))
    return out
''', 'ipin_mod_loop')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == [1.0, 1.0, 2.0, 2.0]


def __test_the_comprehension_iterable_site_configures_it_too__():
    """The walrus-free helper form takes the vector in the same position"""
    ns, dump = _transform(GROUP.replace('-> float', '-> list')
                          .replace('return 1.0', 'return [1.0]')
                          .replace('return 2.0', 'return [2.0]') + SHARED_BODY + '''
def main(r: int):
    return [v for v in f(r)], [v for v in f(r * 1.5)]
''', 'ipin_mod_comp')
    assert "__slot_state·__(__state__, 0, f, ('i',))" in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == ([1.0], [2.0])


UNIFORM_CALLER = '''
def keep(fn):
    return fn

@keep
def f(x):
    return g(x / 2)
'''


def __test_the_uniform_route_carries_the_vector_too__():
    """A decorated callee routes uniform; the binder takes the vector last"""
    ns, dump = _transform(GROUP + UNIFORM_CALLER + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_uniform')
    # The pin position is filled in with None: the binder takes them
    # positionally, and this site has a vector but nothing to pin
    assert "__bind_any·__(__state__, 0, f, None, ('i',))" in dump
    assert '__bind_any·__(__state__, 1, f)' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == (1.0, 2.0)


def __test_the_vector_nests_through_a_chain__():
    """``main -> helper -> inner``: only ``inner`` sees the group"""
    ns, dump = _transform(GROUP + '''
def inner(y):
    return g(y / 2)

def helper(x):
    return inner(x)

def main(r: int):
    return helper(r), helper(r * 1.5)
''', 'ipin_mod_chain')
    # ``helper`` varies with what ITS instance of ``inner`` resolves to, so its
    # vector holds the inner vector, and the inner site reads it out of the slot
    assert "__resolve_slot·__(__state__, 0, helper, (('i',),))" in dump
    assert '__resolve_slot·__(__state__, 1, inner, __state__[0][0])' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == (1.0, 2.0)


PERSISTENT_GROUP = '''
from pynecore import Persistent
from pynecore.core.overload import overload

@overload
def g(x: int) -> float:
    count: Persistent[int] = 0
    count += 1
    return count * 1.0

@overload
def g(x: float) -> float:
    count: Persistent[int] = 0
    count += 1
    return count * 100.0
'''


def __test_each_instance_keeps_its_own_state_across_bars__():
    """The implementation a configured instance picked accumulates on its own"""
    ns, _ = _transform(PERSISTENT_GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_persist')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == (1.0, 100.0)
    assert ns['main'](state, 1) == (2.0, 200.0)
    assert ns['main'](state, 1) == (3.0, 300.0)


# --- the vector is not state the body computed ----------------------------


def __test_the_vector_survives_a_per_bar_redefinition__():
    """``_carry_state`` keeps the vector with the instance it belongs to"""
    # Both contexts are needed for the body to VARY at all -- the anchor the
    # test drives by hand is the int one
    ns, _ = _transform(PERSISTENT_GROUP + UNIFORM_CALLER + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_carry')
    layout = ns['__pyne_slot_layout__']['f']
    parent = _make_state(ns['__pyne_slot_layout__']['main'])
    # The anchor's first bind creates the instance and configures it
    bound = __bind_any__(parent, 0, ns['f'], None, ('i',))
    assert bound(1) == 1.0
    vector = bound.args[0][layout['pin']]
    # A redefinition of the same logical callee misses the identity check and
    # rebinds; the state vector is carried over, and so is what configured it
    __bind_any__(parent, 0, ns['f'], None, ('i',))
    carried = parent[0][1].args[0]
    assert carried is bound.args[0]
    assert carried[layout['pin']] == vector == ('i',)
    assert parent[0][1](1) == 2.0  # the int implementation kept counting


def __test_the_vector_is_excluded_from_the_var_rollback__():
    """The var rollback leaves the slot alone: it is not a bar's own result"""
    ns, _ = _transform(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_rollback')
    layout = ns['__pyne_slot_layout__']['f']
    state = create_root('test·ipin·rollback', layout)
    state[layout['pin']] = ('i',)
    try:
        snapshot = RootVarSnapshot(['test·ipin·rollback'])
        snapshot.save()
        state[layout['pin']] = ('f',)
        snapshot.restore()
        # Untouched by the rollback -- had it been covered, the write above
        # would have been undone back to the snapshotted ('i',)
        assert state[layout['pin']] == ('f',)
    finally:
        discard_root('test·ipin·rollback')


def __test_the_vector_survives_the_mid_bar_builtin_reinit__():
    """A machine born mid-bar is re-initialized in place, vector included

    The same-bar rollback of a loop site re-creates a machine that has no
    bar-start snapshot to return to. What the call site configured is not part
    of that bar's advance, so it has to come through the re-init.
    """
    layout = {'init': ((None,), 7), 'series': (), 'varip': (), 'children': (),
              'compacted': True, 'pin': 0}
    vector = _make_state(layout)
    vector[0] = ('i',)
    vector[1] = 42
    _restore_collected([(vector, layout)], [])
    assert vector[0] == ('i',)
    assert vector[1] == 7


# --- the shapes that configure nothing ------------------------------------


def __test_a_callee_with_no_slot_takes_no_vector__():
    """A layout with no ``pin`` key ignores a vector it is handed anyway"""
    ns, _ = _transform('''
from pynecore import Persistent

def acc(x):
    total: Persistent[int] = 0
    total += x
    return total
''', 'ipin_mod_noslot')
    layout = ns['__pyne_slot_layout__']['acc']
    assert 'pin' not in layout
    parent: list = [None, None]
    bound = __bind_any__(parent, 0, ns['acc'], None, ('i',))
    assert bound(2) == 2


@pytest.mark.parametrize("call", [
    'f(*args)',      # an unpacking hides which position is which
    'f(**kw)',       # ... and which name
    'f(1, 2, 3)',    # more arguments than the callee has parameters
])
def __test_an_unresolvable_call_shape_configures_nothing__(call: str):
    """A shape the type pass cannot instantiate leaves the default in place"""
    tree = ast.parse(GROUP + SHARED_BODY + f'''
args = (1,)
kw = {{}}
z = {call}
''')
    infer_module(tree, 'test')
    site = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'f'][-1]
    assert get_vector(site) is None


def __test_a_cross_module_callee_binds_with_no_vector__():
    """An imported callee was transformed with its own module's contexts only"""
    lib_ns, _ = _transform(GROUP + SHARED_BODY + '''
def main(r: int):
    return f(r), f(r * 1.5)
''', 'ipin_mod_lib')
    layout = lib_ns['__pyne_slot_layout__']['f']
    # Its body varies, so it HAS a slot -- and a caller from another module
    # has no vector to hand it, which is the all-None default: value dispatch
    assert 'pin' in layout
    parent: list = [None, None]
    bound = __bind_any__(parent, 0, lib_ns['f'])
    assert bound(1) == 2.0
    assert bound.args[0][layout['pin']] == (None,)


def __test_a_rebound_callee_gets_no_instance_vector__():
    """
    A call through a rebound name configures nothing, and cannot.

    ``one = two`` makes the ``one`` sites reach ``two``, whose vector has one
    entry per varying site of ITS body. Stamping those sites with the vector of
    the definition named ``one`` writes a one-entry list into the slot ``two``
    indexes twice, and its second site reads past the end. The name is opaque
    to the type pass -- no context, no pin, no vector -- and the values decide,
    which is what the isolation pass's uniform route already assumed.
    """
    source = GROUP + '''
def one(v):
    return g(v)

def two(v):
    return g(v), g(v)

one = two

def main(r: int):
    return one(r), one(r * 1.5), two(r), two(r * 1.5)
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    rebound = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'one']
    assert [get_vector(node) for node in rebound] == [None, None]

    ns, _ = _transform(source, 'instance_pin_rebound')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == ((1.0, 1.0), (2.0, 2.0), (1.0, 1.0), (2.0, 2.0))
