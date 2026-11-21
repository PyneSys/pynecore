"""
A user function's parameters are typed PER CALL SITE.

MEASURED on TradingView: the type of an unannotated parameter at a call site
is JOIN(type of its default, type of the argument), and the body behaves as if
it were instantiated once per distinct parameter-type tuple. The engine walks
one body once per context and keeps the answers apart, so a helper called with
an int and with a float returns an int at the first site and a float at the
second -- and an overload group inside such a helper keeps its pin wherever
every context agrees on it.

The pin is the reason this matters: it is the only place where the static
answer differs from what the values say, and a parameter that stayed UNKNOWN
cost every call site inside the helper its pin.
"""
import ast

import pytest

from pynecore.core.instance_state import _make_state
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.layout_emit import apply_layout
from pynecore.transformers.module_interface import build_interface
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import get_pin, get_pins, get_ty
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout


def _types(source: str, scope: str = '') -> dict[str, str]:
    """Infer a snippet and return one scope's bindings as name -> type."""
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    return {name: binding.ty for name, binding in table.bindings.get(scope, {}).items()}


def _calls(source: str, callee: str) -> tuple[ast.Module, list[ast.Call]]:
    """Infer a snippet and return its tree with every call to one bare name."""
    tree = ast.parse(source)
    infer_module(tree, 'test')
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
    return tree, found


def __test_one_body_two_contexts__():
    """A generic helper is int at the int call site and float at the float one"""
    types = _types(
        'def f(x):\n'
        '    return x + 1\n'
        '\n'
        'a = f(2)\n'
        'b = f(2.0)\n'
    )
    assert types['a'] == 'i'
    assert types['b'] == 'f'


@pytest.mark.parametrize("definition,call,expected", [
    # The JOIN law: an int default with a float argument widens, and so does
    # a float default with an int argument -- the default is a PARTNER of the
    # argument, never a declaration on its own
    ('def f(x=0):\n    return x\n', 'f(1.5)', 'f'),
    ('def f(x=0.0):\n    return x\n', 'f(1)', 'f'),
    ('def f(x):\n    return x\n', 'f(1)', 'i'),
    # ... and where the caller omits the argument, the default IS the value
    ('def f(x=0):\n    return x\n', 'f()', 'i'),
    # An annotation still outranks both halves of the join
    ('def f(x: float = 0):\n    return x\n', 'f(1)', 'f'),
])
def __test_the_join_law__(definition: str, call: str, expected: str):
    """A parameter's type at a call site is JOIN(default, argument)"""
    assert _types(f'{definition}\nz = {call}\n')['z'] == expected


def __test_a_default_alone_still_declares_nothing__():
    """With no call site there is nothing to join the default with"""
    assert _types('def helper(x=0):\n    return x\n', 'helper')['x'] == '?'


def __test_a_keyword_argument_binds_to_its_parameter__():
    """The context tuple follows the parameter, not the argument position"""
    types = _types(
        'def f(a, b):\n'
        '    return a + b\n'
        '\n'
        'x = f(b=1, a=2)\n'
        'y = f(1, b=2.0)\n'
    )
    assert types['x'] == 'i'
    assert types['y'] == 'f'


@pytest.mark.parametrize("call", [
    'f(*args)',      # an unpacking hides which position is which
    'f(**kw)',       # ... and which name
    'f(1, 2, 3)',    # more arguments than the callee has parameters
    'f(wrong=1)',    # a name the callee does not declare
])
def __test_an_unresolvable_shape_falls_back__(call: str):
    """A call shape the analysis cannot describe types the callee alone"""
    types = _types(
        'def f(x):\n'
        '    return x\n'
        '\n'
        'args = (1,)\n'
        'kw = {}\n'
        f'z = {call}\n'
    )
    assert types['z'] == '?'


def __test_nested_contexts_follow_their_caller__():
    """``inner``'s context depends on the context ``helper`` was analysed in"""
    source = (
        'def inner(v):\n'
        '    return v * 2\n'
        '\n'
        'def helper(x):\n'
        '    return inner(x)\n'
        '\n'
        'a = helper(3)\n'
        'b = helper(3.5)\n'
    )
    types = _types(source)
    assert types['a'] == 'i'
    assert types['b'] == 'f'

    table = infer_module(ast.parse(source), 'test')
    inners = {result.params: result.ret for result in table.contexts.values()
              if result.key == 'inner'}
    assert inners == {('i',): 'i', ('f',): 'f'}


def __test_the_canonical_dfs_dependency__():
    """
    A call whose ARGUMENT is another call of the same helper resolves.

    ``momentum_strategy`` (PyneComp's compiled corpus) is the shape: the second
    call's context can only be computed once the first call's context has been
    analysed, which is exactly what walking the callee from the call site buys.
    """
    source = (
        'from pynecore import lib\n'
        '\n'
        'def momentum(price, length):\n'
        '    return price - price / length\n'
        '\n'
        'mom0 = momentum(lib.close, 12)\n'
        'mom1 = momentum(mom0, 1)\n'
    )
    types = _types(source)
    assert types['mom0'] == 'f'
    assert types['mom1'] == 'f'

    table = infer_module(ast.parse(source), 'test')
    contexts = {result.params for result in table.contexts.values()
                if result.key == 'momentum'}
    assert contexts == {('f', 'i')}


def __test_a_forward_reference_resolves_on_one_pass__():
    """A call to a definition further down needs no second walk of the module"""
    types = _types(
        'z = later(2)\n'
        '\n'
        'def later(x):\n'
        '    return x + 1\n'
    )
    assert types['z'] == 'i'


def __test_a_definition_inside_an_if_is_typed__():
    """A ``def`` in a compound statement is a definition like any other"""
    source = (
        'from pynecore import lib\n'
        'if lib.bar_index >= 0:\n'
        '    def hidden(x):\n'
        '        return x * 2\n'
        'z = hidden(3)\n'
    )
    assert _types(source)['z'] == 'i'
    table = infer_module(ast.parse(source), 'test')
    assert table.funcs['hidden'].ret == 'i'


def __test_direct_recursion_stays_unknown_without_looping__():
    """A function calling itself is untypable, and must still settle"""
    types = _types(
        'from pynecore import lib\n'
        'def down(n):\n'
        '    return 1 if n <= 0 else down(n - 1)\n'
        '\n'
        'z = down(3)\n'
    )
    assert types['z'] == '?'


def __test_mutual_recursion_stays_unknown_without_looping__():
    """Two helpers returning each other settle the same way"""
    types = _types(
        'def ping(n):\n'
        '    return pong(n)\n'
        '\n'
        'def pong(n):\n'
        '    return ping(n)\n'
        '\n'
        'z = ping(1)\n'
    )
    assert types['z'] == '?'


def __test_a_recursive_call_is_reported__():
    """The UNKNOWN a re-entrant call produces carries its provenance"""
    tree = ast.parse(
        'def down(n):\n'
        '    return down(n - 1)\n'
        '\n'
        'z = down(3)\n'
    )
    table = infer_module(tree, 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'recursion' in reasons


CROSSING_GROUPS = '''
from pynecore import lib
from pynecore.core.overload import overload
from pynecore.core.pine_udt import udt


@udt
class P:
    a: float = lib.na


@udt
class Q:
    b: int = 0


def main():
    @overload
    def show(this: Q) -> str:
        return lib.string.tostring(this.b)

    @overload
    def get(this: P, key: str):
        return show(Q.new()) if key == 'q' else lib.string.tostring(this.a)

    @overload
    def show(this: P):
        out: list[str] = lib.array.new_string()
        lib.array.push(out, get(this, 'a'))
        return lib.array.join(out, ',')

    s = show(P.new(1.0))
'''


def __test_a_group_call_walks_only_the_implementation_it_selects__():
    """
    Two groups calling each other settle without a stale UNKNOWN.

    ``get`` calls ``show`` with a ``Q``, which selects the implementation declared
    before it. The ``P`` one, declared after ``get`` and calling back into it,
    is not reached from there: walking it while ``get`` is in progress would
    read ``get``'s return as UNKNOWN and keep the report it causes.
    """
    table = infer_module(ast.parse(CROSSING_GROUPS), 'test')
    assert table.diags == []
    assert table.bindings['main']['s'].ty == 's'
    walks = [(result.key, result.params) for result in table.contexts.values()
             if result.key in ('main·show', 'main·get')]
    assert sorted(walks) == sorted(set(walks))
    assert len(walks) == 3


GROUP_RECURSION = '''
from pynecore.core.overload import overload
from pynecore.core.pine_udt import udt


@udt
class A:
    v: int = 0


@udt
class B:
    a: A = None


def main():
    @overload
    def init(this: A) -> A:
        this.v = 1
        return this

    @overload
    def init(this: B) -> B:
        this.a = init(A.new())
        return {self_call}

    @overload
    def init(this: int) -> int:
        return this

    b = init(B.new())
'''


@pytest.mark.parametrize("self_call,recursive", [
    # The B implementation selects itself
    ('init(this)', True),
    # ... and another implementation of the group is an ordinary call
    ('this', False),
])
def __test_a_group_implementation_selecting_itself_is_reported__(self_call: str,
                                                                 recursive: bool):
    """A call the static selection hands to the implementation it is made from is a recursion"""
    table = infer_module(ast.parse(GROUP_RECURSION.format(self_call=self_call)), 'test')
    reasons = [diag.origin.reason for diag in table.diags if diag.origin is not None]
    assert reasons == (['recursion'] if recursive else [])


def __test_a_loop_body_with_a_context_call_converges__():
    """
    A generic helper called from a loop body gets both of the loop's contexts.

    ``total`` starts int and widens to float over the loop, so the helper is
    analysed once per type and the binding it feeds ends up joined.
    """
    source = (
        'from pynecore import lib\n'
        'def twice(v):\n'
        '    return v * 2\n'
        '\n'
        'total = 0\n'
        'seen = 0\n'
        'for _ in range(3):\n'
        '    seen = twice(total)\n'
        '    total = total + 0.5\n'
    )
    types = _types(source)
    assert types['total'] == 'f'
    assert types['seen'] == 'f'

    table = infer_module(ast.parse(source), 'test')
    contexts = {result.params: result.ret for result in table.contexts.values()
                if result.key == 'twice'}
    assert contexts == {('i',): 'i', ('f',): 'f'}


# --- the pin across contexts ---------------------------------------------

GROUP = '''
from pynecore.core.overload import overload

@overload
def g(x: int) -> str:
    return 'int-impl'

@overload
def g(x: float) -> str:
    return 'float-impl'
'''


def __test_agreeing_contexts_keep_the_pin_a_constant__():
    """Two int contexts justify the same pin, so the call site keeps it"""
    _, calls = _calls(GROUP + '''
def wrapper(v):
    return g(v)

a = wrapper(1)
b = wrapper(2)
''', 'g')
    assert len(calls) == 1
    assert get_pin(calls[0]) == 'i=0'
    assert get_pins(calls[0]) is None


def __test_disagreeing_contexts_replace_the_pin_with_a_map__():
    """
    An int and a float context cannot share one pin, so both are recorded.

    The single pin is erased -- emitting it would make the float instance call
    the int implementation -- and the per-context map is what a later pass
    turns into per-instance data.
    """
    source = GROUP + '''
def wrapper(v):
    return g(v)

a = wrapper(1)
b = wrapper(2.0)
'''
    tree, calls = _calls(source, 'g')
    assert get_pin(calls[0]) is None
    pins = get_pins(calls[0])
    assert pins is not None
    assert sorted(pins.values(), key=str) == [None, 'i=0']

    table = infer_module(ast.parse(source), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'context-dependent-pin' in reasons


def __test_helper_context_pin_reported(source: str) -> bool:
    table = infer_module(ast.parse(source), 'test')
    return any(diag.origin is not None and diag.origin.reason == 'context-dependent-pin'
               for diag in table.diags)


def __test_a_delivered_per_context_pin_is_not_reported__():
    """
    The same disagreeing site, its instances created inside ``main``: each call
    site hands its instance the vector entry that holds its own pin, so the int
    instance runs the int implementation and nothing needs annotating
    """
    source = GROUP + '''
def wrapper(v):
    return g(v)

def main():
    a = wrapper(1)
    b = wrapper(2.0)
'''
    _, calls = _calls(source, 'g')
    assert sorted(get_pins(calls[0]).values(), key=str) == [None, 'i=0']
    assert not __test_helper_context_pin_reported(source)


def __test_an_undelivered_pin_is_reported_through_the_callers__():
    """
    An instance is handed its vector only where its caller is handed its own:
    ``holder`` is created by raw calls of the module body, so the ``wrapper``
    instances it creates get nothing, and ``g(v)`` dispatches on its values;
    created from ``main`` instead, every instance down the chain gets its pin
    """
    body = '''
def wrapper(v):
    return g(v)

def holder(v):
    return wrapper(v)
'''
    assert __test_helper_context_pin_reported(GROUP + body + '''
a = holder(1)
b = holder(2.0)
''')
    assert not __test_helper_context_pin_reported(GROUP + body + '''
def main():
    a = holder(1)
    b = holder(2.0)
''')


def __test_an_undelivered_pin_the_values_settle_is_not_reported__():
    """
    A site no instance is handed its pin at, where the values settle every
    context on the implementation its pin does: the object argument rules the
    other implementation out in both passes of the selector, so the int-typed
    argument decides nothing
    """
    source = '''
from pynecore.core.overload import overload
from pynecore.core.pine_udt import udt

@udt
class A:
    v: int = 0

@udt
class B:
    v: int = 0

@overload
def g(x: int, c: A) -> str:
    return 'a'

@overload
def g(x: float, c: B) -> str:
    return 'b'

def wrapper(v, c):
    return g(v, c)

a = wrapper(1, A())
b = wrapper(2.0, B())
'''
    _, calls = _calls(source, 'g')
    assert sorted(get_pins(calls[0]).values(), key=str) == [None, 'i*=0']
    assert not __test_helper_context_pin_reported(source)


@pytest.mark.parametrize('call, of_int, of_float', [
    ('lib.math.max(v, 0)', 'i', 'f'),
    ('lib.math.min(v, 0)', 'i', 'f'),
    ('lib.math.round(v / 7, 1)', 'f', 'f'),
    ('lib.math.round(v)', 'i', 'i'),
])
def __test_one_function_behind_its_overloads_has_nothing_to_pin__(call: str, of_int: str,
                                                                 of_float: str):
    """
    ``math.max`` spells its types as overloads of a single function.

    An int and a float context run the same body, so the site carries no pin,
    and the contexts have nothing to disagree about. Each still types the call
    by its own arguments.
    """
    source = f'''
from pynecore import lib

def wrapper(v):
    return {call}

a = wrapper(1)
b = wrapper(2.0)
'''
    _, calls = _calls(source, call.partition('(')[0])
    assert len(calls) == 1
    assert get_pin(calls[0]) is None
    assert get_pins(calls[0]) is None

    table = infer_module(ast.parse(source), 'test')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert (types['a'], types['b']) == (of_int, of_float)


def __test_a_dispatching_lib_group_keeps_its_pin_per_context__():
    """``ta.highest`` selects an implementation, so its contexts are told apart"""
    source = '''
from pynecore import lib

def wrapper(v):
    return lib.ta.highest(v, 5)

a = wrapper(1)
b = wrapper(2.0)
'''
    _, calls = _calls(source, 'lib.ta.highest')
    assert get_pin(calls[0]) is None
    pins = get_pins(calls[0])
    assert pins is not None
    assert sorted(pins.values()) == ['fi=0', 'ii=0']


def _run(source: str, mod_name: str):
    """Run the slot mini pipeline WITH the type pass and exec the result.

    The type pass sits exactly where the real pipeline puts it: after the
    closure arguments, before the series and isolation passes.

    :param source: Pyne-style module source
    :param mod_name: Unique module name (isolates the overload registry)
    :return: The exec'd module namespace
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
    exec(compile(tree, '<type-context-test>', 'exec'), ns)  # noqa: S102
    return ns


def __test_disagreeing_contexts_still_dispatch_by_value__():
    """
    With no pin to emit, the two instances fall back to the values -- correctly.

    Neither argument is int-TYPED-but-float-VALUED here, so the ordinary
    value-driven selector already agrees with TradingView at both sites. That
    is the fallback ladder: the pin is dropped exactly where it cannot be one
    constant, and nothing regresses.
    """
    ns = _run(GROUP + '''
def wrapper(v):
    return g(v)

def main(r: int):
    return wrapper(r), wrapper(r * 1.0)
''', 'ctx_mod_a')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == ('int-impl', 'float-impl')


def __test_an_agreeing_pin_reaches_the_binder_and_dispatches_on_the_type__():
    """
    The pin an int-only helper earns is what closes the int-division gap.

    ``r / 8`` is int-TYPED with a fractional value, so the values alone would
    pick the float implementation at both sites; the parameter is typed per
    context, both contexts are int, and the pin makes the type decide.
    """
    ns = _run(GROUP + '''
def wrapper(v):
    return g(v)

def main(r: int):
    return wrapper(r / 8), wrapper(r / 7)
''', 'ctx_mod_b')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == ('int-impl', 'int-impl')


def __test_a_call_site_never_instantiates_an_overload_group__():
    """
    A group is per-signature already; the pin is what selects among them.

    Its implementations are analysed once each, under the signature they
    declare -- so however many call sites reach the group, and with whatever
    argument types, the number of contexts does not move.
    """
    one = infer_module(ast.parse(GROUP + '''
def main(r: int):
    return g(r)
'''), 'test')
    many = infer_module(ast.parse(GROUP + '''
def wrapper(v):
    return g(v)

def main(r: int):
    return g(r), g(r * 1.0), wrapper(r), wrapper(r * 1.0)
'''), 'test')
    declared = [result.params for result in one.contexts.values() if result.key == 'g']
    assert sorted(declared) == [('f',), ('i',)]
    assert [result.params for result in many.contexts.values()
            if result.key == 'g'] == declared
    assert one.funcs['g'].ret == 's'


def __test_a_widened_closure_variable_re_types_the_nested_helper__():
    """
    A nested helper is re-analysed when what it closes over widens.

    ``wrapper`` takes no arguments, so its parameter tuple never moves, and the
    calling context's id does not move either while the loop fixpoint widens
    ``total`` from int to float. The types of the enclosing bindings the body
    READS are what tell the two walks apart -- without them the helper keeps
    the pin the first pass justified, and every iteration calls the int
    implementation.

    The re-analysis SUPERSEDES the stale one rather than joining it: the same
    body under the same caller has one answer, the last one.
    """
    source = GROUP + '''
def main(n: int):
    total = 0
    def wrapper():
        return g(total)
    out = []
    for _ in range(n):
        out.append(wrapper())
        total = total + 0.5
    return out
'''
    tree, calls = _calls(source, 'g')
    assert get_pin(calls[0]) is None
    assert get_pins(calls[0]) is None

    table = infer_module(ast.parse(source), 'test')
    assert [result.params for result in table.contexts.values()
            if result.key == 'main·wrapper'] == [()]

    ns = _run(source, 'ctx_mod_closure')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3) == ['int-impl', 'float-impl', 'float-impl']


def __test_a_nested_parameter_does_not_hide_the_wrappers_own_read__():
    """
    ``inner(total)`` declares its OWN ``total``; the wrapper still reads main's.

    The free names of a body used to be one flat loaded-minus-bound set over
    the whole subtree, nested definitions included, so a parameter of a nested
    def cancelled the enclosing read of the same name. The memo key then never
    moved while the loop fixpoint widened ``total``, the first pass's int pin
    stood for good, and every iteration called the int implementation.
    """
    source = GROUP + '''
def main(n: int):
    total = 0
    def wrapper():
        def inner(total):
            return total
        return g(total)
    out = []
    for _ in range(n):
        out.append(wrapper())
        total = total + 0.5
    return out
'''
    tree, calls = _calls(source, 'g')
    assert get_pin(calls[0]) is None
    assert get_pins(calls[0]) is None

    ns = _run(source, 'ctx_mod_nested_param')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3) == ['int-impl', 'float-impl', 'float-impl']


def __test_a_nonlocal_declaration_is_still_a_read_of_the_enclosing_scope__():
    """
    ``nonlocal total`` does not make ``total`` a local of the wrapper.

    The declaration says the opposite: the name lives one scope out, and every
    read of it goes there. Counting it as a binding -- which the rebound-name
    check does, and rightly, because the wrapper can ASSIGN through it --
    subtracted it from the wrapper's free names, so the memo key stood still
    while the loop fixpoint widened ``total`` and the first pass's int pin was
    handed to every iteration.
    """
    source = GROUP + '''
def main(n: int):
    total = 0
    def wrapper():
        nonlocal total
        return g(total)
    out = []
    for _ in range(n):
        out.append(wrapper())
        total = total + 0.5
    return out
'''
    _, calls = _calls(source, 'g')
    assert get_pin(calls[0]) is None
    assert get_pins(calls[0]) is None

    ns = _run(source, 'ctx_mod_nonlocal')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3) == ['int-impl', 'float-impl', 'float-impl']


def __test_a_global_declaration_is_still_a_read_of_the_module_scope__():
    """
    The same holds for ``global``, against the module's own binding.

    Both halves of the declaration have to move, not just the read: ``main``
    widens the MODULE's ``total``, and only a wrapper reading that same
    binding is re-analysed when it does.
    """
    source = GROUP + '''
total = 0

def wrapper():
    global total
    return g(total)

def main(n: int):
    global total
    out = []
    for _ in range(n):
        out.append(wrapper())
        total = total + 0.5
    return out
'''
    _, calls = _calls(source, 'g')
    assert get_pin(calls[0]) is None
    assert get_pins(calls[0]) is None

    ns = _run(source, 'ctx_mod_global')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3) == ['int-impl', 'float-impl', 'float-impl']


def __test_a_global_name_reads_the_module_binding_not_an_intermediate_one__():
    """
    ``global`` skips every scope in between, so the search must too.

    ``outer`` has a float local called ``total``; the wrapper's ``global``
    declaration means the module's int-TYPED ``1 / 2`` all the same. Walking
    the frames outward would stop at the float one and lose the pin, and the
    fractional VALUE would then take the site to the float implementation.
    """
    source = GROUP + '''
total = 1 / 2

def outer(n: int):
    total = n * 1.0
    def wrapper():
        global total
        return g(total)
    return wrapper(), total

def main(n: int):
    return outer(n)
'''
    _, calls = _calls(source, 'g')
    assert get_pin(calls[0]) == 'i=0'

    ns = _run(source, 'ctx_mod_global_skip')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == ('int-impl', 14.0)


def __test_two_call_sites_of_one_closure_keep_their_own_contexts__():
    """
    Two sites are two instantiations, however alike their shapes look.

    ``f`` closes over ``x``, is called once while ``x`` is int-typed and again
    after it has widened. Superseding on everything BUT the call node made the
    second analysis erase the first, leaving one context, no per-instance
    vector, and the int site dispatching on the fractional value 0.5 -- the
    float implementation TradingView never picks there.
    """
    source = GROUP + '''
def main(n: int):
    x = 1 / 2
    def f():
        return g(x)
    first = f()
    x = x + 0.5
    second = f()
    return first, second
'''
    tree, calls = _calls(source, 'g')
    assert get_pin(calls[0]) is None
    assert sorted(str(pin) for pin in (get_pins(calls[0]) or {}).values()) == ['None', 'i=0']

    table = infer_module(ast.parse(source), 'test')
    assert [result.params for result in table.contexts.values()
            if result.key == 'main·f'] == [(), ()]

    ns = _run(source, 'ctx_mod_two_sites')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3) == ('int-impl', 'float-impl')


#: Two implementations that differ only in their container's ELEMENT type.
#: Both parameters head to an object, so anything reading the head alone --
#: the overload pin among them -- cannot tell the two apart.
CONTAINER_GROUP = '''
@overload
def take(xs: list[int], v: int) -> str:
    return g(v / 8)

@overload
def take(xs: list[float], v: int) -> str:
    return g(v / 7)
'''


def __test_every_implementation_of_a_group_is_analysed__():
    """
    Two implementations that look alike from here still get a context each.

    ``list[int]`` and ``list[float]`` are distinct types but the same object
    HEAD, so anything keyed on the head alone cannot tell the two bodies
    apart -- and the second body was never walked, so the pinnable call inside
    it went out with no pin and dispatched on the fractional VALUE.
    """
    source = GROUP + CONTAINER_GROUP + '''
def main(r: int):
    return take([1], r), take([1.0], r)
'''
    _, calls = _calls(source, 'g')
    assert [get_pin(call) for call in calls] == ['i=0', 'i=0']

    table = infer_module(ast.parse(source), 'test')
    assert [result.params for result in table.contexts.values()
            if result.key == 'take'] == [('a:i', 'i'), ('a:f', 'i')]

    ns = _run(source, 'ctx_mod_container')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == ('int-impl', 'int-impl')


def __test_a_rebound_function_name_resolves_to_nothing__():
    """
    A name that is assigned as well as defined calls something else.

    Everything derived from the definition -- its contexts, its pin, its
    instance vector -- would describe a function the call never reaches, so
    the name is opaque and the UNKNOWN says why.
    """
    source = '''
def helper(x):
    return x + 1

def other(x):
    return x + 1.5

helper = other

rebound = helper(2)
plain = other(2)
'''
    assert _types(source)['rebound'] == '?'
    assert _types(source)['plain'] == 'f'

    table = infer_module(ast.parse(source), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'rebound-name' in reasons


#: A group and a plain function to rebind its name to. Module scope binds
#: SEQUENTIALLY, so where the call stands relative to the rebinding decides
#: what it reaches.
REBOUND_GROUP = GROUP + '''
def other(x):
    return 'other'
'''


def __test_a_module_call_above_the_rebinding_reaches_the_definition__():
    """
    ``result = g(1 / 2)`` above ``g = other`` still calls the group.

    The module body runs top to bottom, so the name holds the definition when
    the call executes. Treating the rebinding as shadowing everywhere lost the
    pin, and the site then dispatched on the VALUE 0.5 into the float
    implementation.
    """
    source = REBOUND_GROUP + '''
result = g(1 / 2)
g = other
'''
    _, calls = _calls(source, 'g')
    assert get_pin(calls[0]) == 'i=0'

    table = infer_module(ast.parse(source), 'test')
    # The call site itself is clean; what is reported is the alias, since a
    # function read as a value is no Pine value
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['function-value']

    ns = _run(source, 'ctx_mod_before_rebind')
    assert ns['result'] == 'int-impl'


@pytest.mark.parametrize("body,name", [
    # Below the rebinding the name holds the assigned value
    ("g = other\nresult = g(1 / 2)\n", 'module-after'),
    # A rebinding standing BEFORE the loop has run by the time the loop does
    ("g = other\nresult = ''\nfor _ in range(1):\n    result = g(1 / 2)\n",
     'module-before-loop'),
    # ... and one inside the loop body has run by the second iteration, even
    # though it stands below the call
    ("result = ''\nfor _ in range(1):\n    result = g(1 / 2)\n    g = other\n",
     'module-inside-loop'),
    # The back edge of ANY enclosing loop counts, at whatever depth
    ("result = ''\nfor _ in range(1):\n    for _ in range(1):\n"
     "        result = g(1 / 2)\n    g = other\n", 'module-outer-loop'),
    # Inside a FUNCTION an assignment makes the name local to the whole body,
    # so the call above it never reaches the definition at all
    ("def main(n: int):\n    result = g(1 / 2)\n    g = other\n    return result\n",
     'function-scope'),
    # ... and a function-scope loop does not change that either
    ("def main(n: int):\n    result = ''\n    for _ in range(n):\n"
     "        result = g(1 / 2)\n    g = other\n    return result\n",
     'function-scope-loop'),
])
def __test_a_rebinding_the_call_cannot_outrun_still_shadows__(body: str, name: str):
    """ Only a module-level call written above the rebinding gets through """
    table = infer_module(ast.parse(REBOUND_GROUP + body), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'rebound-name' in reasons, name
    assert [call for call in table.calls if call.callee == 'g'] == []


@pytest.mark.parametrize("body,name", [
    # A module-level loop COMPLETES before the statement below it runs, so its
    # back edge cannot carry a rebinding written outside it back into the body
    ("result = ''\nfor _ in range(1):\n    result = g(1 / 2)\ng = other\n", 'module-loop'),
    ("result = ''\nn = 1\nwhile n > 0:\n    result = g(1 / 2)\n    n = n - 1\ng = other\n",
     'module-while'),
    # Nested loops, with the rebinding outside both of them
    ("result = ''\nfor _ in range(1):\n    for _ in range(1):\n"
     "        result = g(1 / 2)\ng = other\n", 'module-nested-loops'),
    # A rebinding inside a LATER loop is still below every call of this one
    ("result = ''\nfor _ in range(1):\n    result = g(1 / 2)\n"
     "for _ in range(1):\n    g = other\n", 'module-later-loop'),
])
def __test_a_module_loop_completes_before_the_rebinding__(body: str, name: str):
    """
    A loop above the rebinding calls the definition on every iteration.

    Rejecting every call made inside a loop treated the back edge as if it
    reached the whole module body: it does not, it only reaches the loop's own
    statements. The site lost its pin over that and dispatched on the
    fractional VALUE of ``1 / 2``, which is the float implementation.
    """
    source = REBOUND_GROUP + body
    _, calls = _calls(source, 'g')
    assert [get_pin(call) for call in calls] == ['i=0'], name

    table = infer_module(ast.parse(source), 'test')
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == ['function-value'], name


def __test_a_module_loop_above_the_rebinding_runs_the_definition__():
    """ The loop's calls reach the group through the pipeline too """
    ns = _run(REBOUND_GROUP + '''
result = ''
for _ in range(1):
    result = g(1 / 2)
g = other
''', 'ctx_mod_loop_before_rebind')
    assert ns['result'] == 'int-impl'


@pytest.mark.parametrize("body,name", [
    # The rebinding stands ABOVE the call and still never runs with it: one
    # branch of an ``if`` executes, the other does not
    ("flag = True\nresult = ''\nif flag:\n    g = other\nelse:\n    result = g(1 / 2)\n",
     'if-else'),
    # ``elif`` is an ``if`` nested in the outer one's ``orelse``, so the same
    # first-divergence test separates it
    ("flag = 1\nresult = ''\nif flag == 0:\n    g = other\n"
     "elif flag == 1:\n    result = g(1 / 2)\n", 'elif'),
    # One ``match`` case runs and the others do not
    ("result = ''\nmatch 1:\n    case 0:\n        g = other\n"
     "    case _:\n        result = g(1 / 2)\n", 'match'),
    # A capture in a case PATTERN binds only when that case is the one taken:
    # the pattern that did not match bound nothing, so the fallback case's
    # call reaches the definition exactly as it would past an empty case body
    ("subject = []\nresult = ''\nmatch subject:\n    case [g]:\n        pass\n"
     "    case _:\n        result = g(1 / 2)\n", 'match-pattern-capture'),
])
def __test_a_rebinding_in_a_branch_the_call_cannot_share_reaches_nothing__(body: str, name: str):
    """
    A rebinding the call is branched away from leaves the call alone.

    Source order says the assignment comes first; control flow says no pass of
    the module runs both. Reading the position alone treated the call as
    shadowed, and the site lost its pin and dispatched on the fractional VALUE
    of ``1 / 2``, which is the float implementation.
    """
    source = REBOUND_GROUP + body
    _, calls = _calls(source, 'g')
    assert [get_pin(call) for call in calls] == ['i=0'], name

    table = infer_module(ast.parse(source), 'test')
    # The call site is clean; what is reported is the alias, a function read
    # as a value -- where there is one
    expected = ['function-value'] if 'g = other' in body else []
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == expected, name


@pytest.mark.parametrize("body,name", [
    # A ``try`` body and its handler are NOT exclusive: the body runs until it
    # raises, so it may well have rebound the name before the handler starts
    ("result = ''\ntry:\n    g = other\nexcept ValueError:\n    result = g(1 / 2)\n",
     'try-handler'),
    # Nor are a loop's body and its ``else``, which both run in the ordinary case
    ("result = ''\nfor _ in range(1):\n    g = other\nelse:\n    result = g(1 / 2)\n",
     'for-else'),
    # Exclusive branches INSIDE a loop are only exclusive per iteration: the
    # next one is free to take the branch this one did not
    ("flag = True\nresult = ''\nfor _ in range(2):\n    if flag:\n        g = other\n"
     "    else:\n        result = g(1 / 2)\n", 'if-else-in-loop'),
    # Below the whole ``if`` there is no branch left to be separated by
    ("flag = True\nif flag:\n    g = other\nresult = g(1 / 2)\n", 'after-the-if'),
    # A capture and the call in the SAME case: the pattern binds before the
    # body it belongs to runs
    ("subject = [1]\nresult = ''\nmatch subject:\n    case [g]:\n"
     "        result = g(1 / 2)\n    case _:\n        pass\n", 'match-same-case'),
    # A case GUARD runs and can still fail, handing the subject to the next
    # case, so the walrus in it has landed by the time that case's call does
    ("subject = 1\nresult = ''\nmatch subject:\n    case 0 if (g := other):\n        pass\n"
     "    case _:\n        result = g(1 / 2)\n", 'match-case-guard'),
    # A capture in a GUARDED case's pattern outlives the guard: the pattern
    # matched and bound it, and only the guard then sent the subject on to the
    # case below, which finds the name already rebound
    ("subject = [other]\nresult = ''\nmatch subject:\n    case [g] if False:\n        pass\n"
     "    case _:\n        result = g(1 / 2)\n", 'match-guarded-capture'),
])
def __test_a_rebinding_the_branches_do_not_separate_still_shadows__(body: str, name: str):
    """ Only branches ONE pass cannot both take let the call through """
    table = infer_module(ast.parse(REBOUND_GROUP + body), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'rebound-name' in reasons, name
    assert [call for call in table.calls if call.callee == 'g'] == []


def __test_the_branch_the_rebinding_missed_runs_the_definition__():
    """ The else branch reaches the group through the pipeline too """
    ns = _run(REBOUND_GROUP + '''
flag = False
result = ''
if flag:
    g = other
else:
    result = g(1 / 2)
''', 'ctx_mod_exclusive_branch')
    assert ns['result'] == 'int-impl'


def __test_a_pattern_that_did_not_match_runs_the_definition__():
    """
    The fallback case reaches the group through the pipeline too.

    ``[]`` has no first element, so ``case [g]`` fails and binds nothing --
    the name still holds the definition when the fallback case calls it. The
    capture used to be read on the ``match`` statement's own path rather than
    its case's, which made it look like a rebinding no branch keeps away from,
    and the site dispatched on the fractional VALUE of ``1 / 2``.
    """
    ns = _run(REBOUND_GROUP + '''
subject = []
result = ''
match subject:
    case [g]:
        pass
    case _:
        result = g(1 / 2)
''', 'ctx_mod_match_pattern')
    assert ns['result'] == 'int-impl'


#: The group, a second group returning a NUMBER, and a float-valued function
#: to capture. What the inner group returns is what types the outer one's
#: argument, so a capture that redirects it moves the outer pin too.
GUARDED_GROUP = GROUP + '''
@overload
def h(x: int) -> int:
    return 1

@overload
def h(x: float) -> float:
    return 1.0


def alt(x):
    return 2.5
'''


def __test_a_guarded_pattern_keeps_its_captures_for_the_later_cases__():
    """
    A failed GUARD does not undo the bindings its pattern already made.

    ``case [h] if False`` matches the subject and binds ``h`` to the captured
    function; only then does the guard fail, and Python hands the subject to
    the next case with that capture still in place. Indexing the pattern on
    the case's own exclusive path claimed the opposite -- the fallback case's
    calls looked like they still reached the definition, kept the int pin and
    returned the int implementation of a call the runtime had redirected to a
    function whose 2.5 the values dispatch into the float one.
    """
    source = GUARDED_GROUP + '''
subject = [alt]
result = ()
match subject:
    case [h] if False:
        pass
    case _:
        result = (h(1 / 2), g(h(1 / 2)))
'''
    _, inner = _calls(source, 'h')
    assert [get_pin(call) for call in inner] == [None, None]
    _, outer = _calls(source, 'g')
    assert [get_pin(call) for call in outer] == [None]

    table = infer_module(ast.parse(source), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'rebound-name' in reasons

    ns = _run(source, 'ctx_mod_guarded_capture')
    assert ns['result'] == (2.5, 'float-impl')


def __test_a_global_declaration_alone_is_not_a_rebinding__():
    """
    ``global g`` stores nothing, so the call still reaches the group.

    The declaration only says WHERE the name lives. Counting it as a binding
    made every body that declares one lose the group it was declaring, and the
    call inside dispatched on the fractional VALUE of ``1 / 2``.
    """
    source = REBOUND_GROUP + '''
def main(n: int):
    global g
    return g(1 / 2)
'''
    _, calls = _calls(source, 'g')
    assert [get_pin(call) for call in calls] == ['i=0']

    table = infer_module(ast.parse(source), 'test')
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == []

    ns = _run(source, 'ctx_mod_global_decl')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 1) == 'int-impl'


def __test_a_store_through_a_global_declaration_still_shadows__():
    """
    ``global g`` followed by ``g = other`` rebinds the name after all.

    The store is what the declaration sends to the module scope, and it can
    land at any time, so the body's own calls cannot count on the definition.
    """
    table = infer_module(ast.parse(REBOUND_GROUP + '''
def main(n: int):
    global g
    g = other
    return g(1 / 2)
'''), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'rebound-name' in reasons
    assert [call for call in table.calls if call.callee == 'g'] == []


#: An overload group in a FUNCTION scope, for the ``nonlocal`` half of the
#: same law: the declaration names the enclosing scope's binding, nothing more.
NESTED_GROUP = '''
from pynecore.core.overload import overload

def other(x):
    return 'other'

def outer(n: int):
    @overload
    def g(x: int) -> str:
        return 'int-impl'

    @overload
    def g(x: float) -> str:
        return 'float-impl'

    def inner():
        nonlocal g
%s        return g(1 / 2)

    return inner()
'''


def __test_a_nonlocal_declaration_alone_is_not_a_rebinding__():
    """ ``nonlocal g`` alone leaves the enclosing group reachable """
    source = NESTED_GROUP % ''
    _, calls = _calls(source, 'g')
    assert [get_pin(call) for call in calls] == ['i=0']

    table = infer_module(ast.parse(source), 'test')
    assert [diag.origin.reason for diag in table.diags
            if diag.origin is not None] == []


def __test_a_store_through_a_nonlocal_declaration_still_shadows__():
    """ ``nonlocal g`` followed by ``g = other`` rebinds the name after all """
    table = infer_module(ast.parse(NESTED_GROUP % '        g = other\n'), 'test')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'rebound-name' in reasons
    assert [call for call in table.calls if call.callee == 'g'] == []


GENERIC_GROUP = '''
from pynecore import lib
from pynecore.core.overload import overload
from pynecore.core.pine_method import method, method_call
from pynecore.core.pine_udt import udt


@udt
class Span:
    low: float = lib.na
    high: float = lib.na


def main():
    @overload
    def wider(x, y, up):
        return x if up else y

    @overload
    @method
    def wider(this: Span, up):
        return this.high if up else this.low

    a = wider(1, 2, True)
    b = wider(1.5, 2.5, False)
    c = wider(Span.new(1.0, 2.0), True)
    d = method_call(wider, Span.new(1.0, 2.0), False)
'''


def __test_an_implementation_without_annotations_is_typed_per_call__():
    """
    An implementation whose parameters declare no type is instantiated by each call
    that reaches it alone, under the arguments that call binds -- Pine's rule for
    any function without one -- and the call evaluates to that instance's return.
    """
    table = infer_module(ast.parse(GENERIC_GROUP), 'test')
    assert table.diags == []
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert (types['a'], types['b'], types['c'], types['d']) == ('i', 'f', 'f', 'f')


def __test_each_argument_shape_is_its_own_instance__():
    """Two calls with different argument types instantiate the implementation twice"""
    table = infer_module(ast.parse(GENERIC_GROUP), 'test')
    contexts = sorted(result.params for result in table.contexts.values()
                      if result.key == 'main·wider')
    assert contexts == [('f', 'f', 'b'), ('i', 'i', 'b'), ('o:test#Span', 'b')]
    # The group's own shape keeps an instantiated implementation's return open:
    # no single call's answer stands for the others
    rets = [impl.ret for impl in table.groups['main·wider']]
    assert rets == ['?', '?']
    assert [impl.bare for impl in table.groups['main·wider']] == ['111', '01']


UNCALLED_GENERIC = '''
from pynecore.core.overload import overload


def main():
    @overload
    def pick(x, y, z):
        return x + y + z

    @overload
    def pick(x: int):
        return x

    a = pick(1)
'''


def __test_a_generic_implementation_nothing_calls_is_not_reported__():
    """
    An implementation no call selects has no instance: its body is typed the way
    its own definition describes it, which keeps the code it emits, and the gaps of
    its untyped parameters are not reported.
    """
    tree = ast.parse(UNCALLED_GENERIC)
    table = infer_module(tree, 'test')
    assert table.diags == []
    contexts = sorted(result.params for result in table.contexts.values()
                      if result.key == 'main·pick')
    assert contexts == [('?', '?', '?'), ('i',)]
    total = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.BinOp) and isinstance(node.left, ast.BinOp))
    assert get_ty(total) == '?'


EXPORTED_GENERIC = '''
from pynecore.core.overload import overload
from pynecore.core.pine_export import export


@export
@overload
def pick(x, y):
    return x + y


@export
@overload
def pick(x: int):
    return x
'''


def __test_an_exported_implementation_keeps_its_declaration_context__():
    """
    An exported implementation publishes the shape its own definition describes:
    it is typed once, in its declaration context, and the published group does not
    carry which of its positions are unannotated.
    """
    tree = ast.parse(EXPORTED_GENERIC)
    table = infer_module(tree, 'test')
    assert [diag.code for diag in table.diags] == ['E0012', 'E0012']
    contexts = sorted(result.params for result in table.contexts.values())
    assert contexts == [('?', '?'), ('i',)]
    assert [impl.bare for impl in table.groups['pick']] == ['11', '0']
    published = build_interface(tree, table, 'exported_generic.py').exports['pick']
    assert [impl.bare for impl in published.impls] == ['', '']


OWN_EXPORT_BY_NAME = '''
from pynecore.core.pine_export import Exported, export
from pynecore import lib
from pynecore.core.pine_method import method, method_call

scaled = Exported()


@lib.script.library("own_export_by_name")
def main():
    @export
    @method
    def scaled(arr: list[float], type__ren__: float):
        return lib.array.get(arr, 0) * type__ren__

    @export
    def twice(arr: list[float]):
        out = method_call('scaled', arr, {keyword}=2.0)
        return out
'''


@pytest.mark.parametrize("keyword,expected,codes", [
    # Pine's spelling of a parameter the library declares renamed: the proxy
    # hands it to the renamed parameter (``core/pine_method._adapt_exported_kwargs``)
    ('type', 'f', []),
    ('type__ren__', 'f', []),
    # A name the definition declares under neither spelling is still no parameter
    ('kind', '?', ['E0063']),
])
def __test_an_own_export_called_by_name_binds_a_renamed_keyword__(keyword: str, expected: str,
                                                                  codes: list[str]):
    """A keyword of a call by name binds the parameter the proxy hands it to"""
    table = infer_module(ast.parse(OWN_EXPORT_BY_NAME.format(keyword=keyword)), 'test')
    assert [diag.code for diag in table.diags] == codes
    assert table.bindings['main·twice']['out'].ty == expected
