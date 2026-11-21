"""
The static overload pin: TradingView resolves an overload from the TYPE.

Pine's ``int`` is a static type only, so ``14 / 8`` is int-TYPED while its
value is 1.75. A dispatcher that looks at values therefore widens such an
argument to the float implementation, while TradingView picks the int one --
the single divergence the measured 16-case table in
``work/tv-int-tipus-reverse-engineering.md`` records (it stood at 14/16).

The pin closes it without duplicating the selector: the type pass writes one
type character per positional argument onto the call site, the binder turns
those characters back into witness values, and the ORDINARY ``_select`` runs
on them once per anchor. Static and dynamic dispatch are the same code, so
they cannot drift apart.
"""
import ast

import pytest

from typing import Any, Optional

from pynecore import lib
from pynecore.core.instance_state import _make_state
from pynecore.core.overload import (  # noqa: internal API
    Implementation, _anchored, _check_type, _select, _PIN_WITNESSES,
)
from pynecore.errorcodes import WrongOperandTypeError
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.layout_emit import apply_layout
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.lib_registry import lib_types
from pynecore.transformers.pine_type_infer import _lib_impl_sigs  # noqa: internal API
from pynecore.transformers.pine_type_rules import (
    FIT_OMISSIBLE, FIT_REQUIRED, INT, ImplSig, NONE_DEFAULT, OBJECT, PIN_WITNESSED, TYPELESS,
    annotation_accepts, annotation_takes_none, class_id, default_fit, get_pin, get_ty,
    object_ty, overload_bind, overload_binding, overload_candidates, overload_choice,
    overload_pick, pin_index, pine_choice, resolve_pin,
)
from pynecore.types.base import StrLiteral
from pynecore.types.matrix import Matrix
from pynecore.types.pine_types import PyneBool, PyneFloat, PyneInt, PyneStr
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout
from pynecore.types.box import Box
from pynecore.types.color import Color
from pynecore.types.line import Line
from pynecore.types.na import NA

# The reverse-engineering probe (``ovl2.pine``, FX:EURUSD@60) AS COMPILED: this
# is PyneComp's output verbatim, with only the measured argument substituted.
# The compiled shape is what has to be typed, and it differs from hand-written
# Pyne in ways that decide the answer -- the inputs arrive as unannotated
# parameter DEFAULTS, and a Pine cast arrives as ``cast_int``. ``z`` is 0 on
# every bar but not statically so, which is what stops the folder.
MEASURED_SETUP = '''
from pynecore.core.overload import overload
from pynecore.core.pine_cast import cast_int
from pynecore.lib import bar_index, close, input, math, na, script, time


@script.indicator("ovl2")
def main(
    R=input.int(14)
):
    @overload
    def f(x: int):
        return 1.0

    @overload
    def f(x: float):
        return 2.0

    z = 0 if bar_index >= 0 else 1
    na_i: int = na(int)
    na_f: float = na(float)
    result = f({expression})
'''


def _call_site(source: str, callee: str = 'f') -> tuple[ast.Call, list]:
    """Infer a snippet and return one call's node and the table's call sites.

    Import normalization runs first, as it does in the pipeline: it is what
    turns ``bar_index`` and ``math.round`` into the ``lib.*`` spellings the
    registry is keyed by.

    :param source: Module source to infer.
    :param callee: Name of the call to return the node of (last occurrence).
    :return: (the call node, every recorded call site)
    """
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    table = infer_module(tree, 'test')
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
    return found[-1], table.calls


# Every row is MEASURED on TradingView; the type is what the chart reported for
# the argument, and the implementation follows from it.
@pytest.mark.parametrize("expression,expected", [
    ('R + z', 'i'),
    # The two rows that used to fail: int / int keeps the int TYPE while the
    # value goes fractional (1.75) or stays exact (2.0) -- neither is a float
    ('(R + z) / 8', 'i'),
    ('(R + z) / 7', 'i'),
    ('na_i', 'i'),
    ('na_f', 'f'),
    ('bar_index', 'i'),
    ('math.round(close)', 'i'),
    ('math.round(close, 2)', 'f'),
    ('math.floor(close)', 'i'),
    ('time', 'i'),
    ('(R + z) * 1.0', 'f'),
    ('cast_int(close)', 'i'),
    ('(R + z) % 4', 'i'),
    ('-(R + z)', 'i'),
    ('math.max(R + z, 2)', 'i'),
    ('math.abs(R + z)', 'i'),
])
def __test_the_measured_overload_table__(expression: str, expected: str):
    """ The argument carries TradingView's type, and the pin names its implementation """
    node, calls = _call_site(MEASURED_SETUP.format(expression=expression))
    assert get_ty(node.args[0]) == expected, expression
    pin = [call.pin for call in calls if call.callee == 'f'][-1]
    # The pin names the implementation of that type by its registration index
    assert pin == ('i=0' if expected == 'i' else 'f=1'), expression
    assert get_pin(node) == pin


def __test_only_an_overload_group_is_pinned__():
    """ A single implementation has nothing to choose between """
    node, calls = _call_site('''
from pynecore import lib

def g(x: int) -> float:
    return 1.0

R = lib.input.int(14)
result = g(R / 8)
''', 'g')
    assert get_pin(node) is None
    assert [call.pin for call in calls if call.callee == 'g'] == [None]


def __test_a_lib_overload_group_is_pinned__():
    """ The registry's overload groups are pinnable the same way """
    node, _ = _call_site('''
from pynecore import lib
R = lib.input.int(14)
result = lib.ta.highest(R / 8)
''', 'lib.ta.highest')
    # ... and name the implementation they resolve to: the length-only one
    assert get_pin(node) == 'i=1'


@pytest.mark.parametrize("call,pin", [
    ('f(R * 1.0)', 'f=1'),     # a float: its implementation, by its index
    ('f(x=R / 8)', '=0'),       # a keyword is decided by its type too
    ('f(*args)', None),         # an unpacking hides which position is which
    ('f(array.new_float(1))', None),  # no implementation takes an array
    ('f(first)', 'f=1'),        # the first element of a float array is a float
])
def __test_the_unpinnable_shapes__(call: str, pin: str | None):
    """ A shape the static types do not decide is declined, never guessed """
    node, _ = _call_site(MEASURED_SETUP.replace(
        'import bar_index', 'import array, bar_index').format(expression='R').replace(
        'result = f(R)',
        'args = [1]\n'
        '    first = array.first(array.new_float(1))\n'
        f'    result = {call}'))
    assert get_pin(node) == pin


# --- emission ------------------------------------------------------------


def _transform(source: str, mod_name: str) -> tuple[dict, str]:
    """Run the slot mini pipeline WITH the type pass and exec the result.

    The type pass sits exactly where the real pipeline puts it: after the
    closure arguments, before the series and isolation passes.

    :param source: Pyne-style module source.
    :param mod_name: Unique module name (isolates the overload registry).
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
    exec(compile(tree, '<overload-pin-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


PIN_SRC = '''
from pynecore.core.overload import overload

@overload
def f(x: int) -> float:
    return 1.0

@overload
def f(x: float) -> float:
    return 2.0
'''


def __test_the_pin_reaches_the_binder__():
    """ Each site carries the type characters and the index of its implementation """
    _, dump = _transform(PIN_SRC + '''
def main(r: int):
    return f(r / 8), f(r * 1.0)
''', 'pin_mod_a')
    assert "__bind_any·__(__state__, 0, f, 'i=0')" in dump
    assert "__bind_any·__(__state__, 1, f, 'f=1')" in dump


def __test_the_loop_and_comprehension_forms_carry_it_too__():
    """ The helper-only shapes take the pin in the same trailing position """
    _, dump = _transform(PIN_SRC + '''
def main(r: int):
    for _ in range(2):
        f(r / 8)
    return [f(r / 8) for _ in [0]]
''', 'pin_mod_b')
    assert "__bind_loop·__(__state__, 0, f, 'i=0')" in dump
    assert "__bind_loop·__(__state__, 1, f, 'i=0')" in dump

    # A comprehension ITERABLE forbids the walrus guard, so the whole site
    # folds into the slot helper -- which takes the pin in that position too
    _, dump = _transform(PIN_SRC.replace('-> float', '-> list')
                         .replace('return 1.0', 'return [1.0]')
                         .replace('return 2.0', 'return [2.0]') + '''
def main(r: int):
    return [v for v in f(r / 8)]
''', 'pin_mod_b2')
    assert "__bind_slot·__(__state__, 0, f, 'i=0')" in dump


# --- runtime -------------------------------------------------------------


def __test_the_pin_dispatches_on_the_type_not_the_value__():
    """ ``r / 8`` is int-typed with a fractional value: the int one wins """
    ns, _ = _transform(PIN_SRC + '''
def main(r: int):
    return f(r / 8), f(r * 1.0), f(r + 1)
''', 'pin_mod_c')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (1.0, 2.0, 1.0)


def __test_the_pin_can_be_switched_off__(monkeypatch):
    """ ``PYNE_NO_TYPE_PIN=1`` dispatches from the values alone again """
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    ns, _ = _transform(PIN_SRC + '''
def main(r: int):
    return f(r / 8), f(r * 1.0), f(r + 1)
''', 'pin_mod_d')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (2.0, 2.0, 1.0)


def __test_the_pinned_anchor_keeps_its_instance_state__():
    """ A pinned anchor binds the implementation's state vector like any other """
    ns, _ = _transform('''
from pynecore import Persistent
from pynecore.core.overload import overload

@overload
def acc(x: int) -> int:
    total: Persistent[int] = 0
    total += 1
    return total

@overload
def acc(x: str) -> int:
    return -1

def main(r: int):
    return acc(r / 8)
''', 'pin_mod_e')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == 1
    assert ns['main'](state, 14) == 2
    assert ns['main'](state, 14) == 3


DISAGREEING_GROUP = '''
from pynecore.core.overload import overload

@overload
def h(x: int) -> int:
    return 1

@overload
def h(x: float) -> float:
    return 1.0
'''


def __test_a_group_types_a_call_only_where_it_agrees__():
    """
    A group whose implementations return different types is UNKNOWN.

    Pine resolves the overload statically, so exactly one return IS the call's
    type -- but this pass does not run the selector, so it cannot say which.
    Joining them instead would produce a float where TradingView has an int,
    and that guess would then decide things (see the test below).
    """
    agreeing = infer_module(ast.parse(DISAGREEING_GROUP.replace('-> int', '-> float')
                                      .replace('return 1\n', 'return 1.0\n')), 'test')
    assert agreeing.funcs['h'].ret == 'f'
    assert infer_module(ast.parse(DISAGREEING_GROUP), 'test').funcs['h'].ret == '?'


def __test_a_pinned_group_feeds_the_outer_pin_with_the_selected_return__():
    """
    A SELECTED return may reach an outer pin; a guessed one may not.

    ``g(h(r), r)``: TradingView resolves ``h`` to its int implementation, so
    the first argument of ``g`` is an int and ``g`` resolves to ``(int, int)``.
    The inner site is pinned ``'i'``, which is the selector's exact pass having
    already named that implementation -- so its int return is the call's type,
    and the outer pin follows from a decision, not from a join. A JOINED ``h``
    would make the argument a float, pin ``g`` as ``'fi'`` and select the
    implementation NEITHER TradingView nor the value-driven dispatch picks.
    """
    source = DISAGREEING_GROUP + '''
@overload
def g(a: int, b: int) -> str:
    return 'A'

@overload
def g(a: float, b: int) -> str:
    return 'B'

def main(r: int):
    return g(h(r), r)
'''
    table = infer_module(ast.parse(source), 'test')
    assert [call.pin for call in table.calls if call.callee == 'h'] == ['i=0']
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['i']
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['ii=0']

    ns, _ = _transform(source, 'pin_mod_join')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == 'A'


def __test_a_later_pass_erases_the_pin_it_no_longer_stands_behind__():
    """
    The loop fixpoint walks a body twice; the second walk decides.

    ``total`` starts int and widens to float over the loop, so the first walk
    pins the int implementation and the second the float one: a stamp left
    over from the first would emit a pin the inference itself rejected, and the
    site would keep calling the int implementation for the rest of the run.
    """
    source = '''
from pynecore.core.overload import overload

@overload
def g(x: int) -> str:
    return 'int-impl'

@overload
def g(x: float) -> str:
    return 'float-impl'

def main(n: int):
    total = 0
    out = []
    for _ in range(n):
        out.append(g(total))
        total = total + 0.5
    return out
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    # The last walk is the verdict, and the node has to carry that one
    assert [call.pin for call in table.calls if call.callee == 'g'][-1] == 'f=1'
    call = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'g'][0]
    assert get_pin(call) == 'f=1'

    ns, _ = _transform(source, 'pin_mod_loop')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    # The float-typed ``total`` runs the float implementation from the first bar,
    # while it still holds the Python int 0
    assert ns['main'](state, 3) == ['float-impl', 'float-impl', 'float-impl']


def _witness_group():
    """A two-implementation group built without the decorator, for the guards."""
    def wide(x: float) -> str:
        return 'float'
    return [Implementation(wide)]


def __test_the_pinned_route_falls_back_to_the_values__():
    """ A shape the pin was not computed for decides the old way """
    impls = _witness_group()
    bound = _anchored(impls, 'test.guard', pin='i=0')
    # The only implementation takes the int witness by widening
    assert bound(1) == 'float'
    # A keyword spelling and a different arity are not that shape
    assert bound(x=1) == 'float'
    with pytest.raises(WrongOperandTypeError):
        bound(1, 2)

    def narrow(x: int) -> str:
        return 'int'
    impls.append(Implementation(narrow))
    # The group grew after the bind, so the pin's answer is stale: the values
    # decide again, and they find the implementation that was not there yet
    assert bound(1) == 'int'


def __test_an_unresolvable_pin_leaves_the_dispatch_alone__():
    """ A pin no implementation answers is dropped, not raised on """
    bound = _anchored(_witness_group(), 'test.unresolvable', pin='s')
    assert bound(1.0) == 'float'


# --- the pin as the selector's exact pass --------------------------------

#: Two groups composed: ``h`` returns a different type per implementation, so
#: the group itself has no type -- only the SELECTED implementation does.
COMPOSED_GROUPS = '''
from pynecore.core.overload import overload

@overload
def h(x: int) -> int:
    return x / 2

@overload
def h(x: float) -> float:
    return x / 2

@overload
def g(x: int) -> str:
    return 'int-impl'

@overload
def g(x: float) -> str:
    return 'float-impl'
'''


def __test_a_pinned_call_is_typed_by_the_implementation_it_selects__():
    """
    A pin has already named an implementation, so its return types the call.

    ``h(1)`` pins ``'i'``, and that pin IS the exact pass of the runtime
    selector: the int implementation runs, so the call is int-typed however
    much the two implementations disagree. Reading the GROUP's type there left
    ``h(1)`` UNKNOWN, ``g(h(1))`` unpinned, and the outer site dispatching on
    the value ``h`` produces -- 0.5, the float implementation TradingView
    never picks.
    """
    table = infer_module(ast.parse(COMPOSED_GROUPS + '''
def main(r: int):
    return g(h(1))
'''), 'test')
    assert table.funcs['h'].ret == '?'
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['i']
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['i=0']


def __test_the_composed_pin_reaches_the_int_implementation__():
    """ The composed groups run the way the pins say, through the pipeline """
    ns, _ = _transform(COMPOSED_GROUPS + '''
def main(r: int):
    return h(1), g(h(1))
''', 'pin_mod_composed')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (0.5, 'int-impl')


@pytest.mark.parametrize("impls,pin,expected", [
    # The implementation whose parameters ARE the pin, in either order
    ([ImplSig(('i',), 1, False, 'i', 'n'), ImplSig(('f',), 1, False, 'f', 'n')], 'i', 'i'),
    ([ImplSig(('f',), 1, False, 'f', 'n'), ImplSig(('i',), 1, False, 'i', 'n')], 'i', 'i'),
    ([ImplSig(('i', 'i'), 2, False, 'A', 'nn'),
      ImplSig(('f', 'i'), 2, False, 'B', 'nn')], 'ii', 'A'),
    # No exact match: the exact pass answers nothing, and the widening pass that
    # takes the int for the float is ``resolve_pin``'s (``RESOLVED_PINS`` below)
    ([ImplSig(('f',), 1, False, 'f', 'n')], 'i', None),
    # An earlier implementation that takes anything would win on declaration
    # order, so the site is unanswerable rather than overruled
    ([ImplSig(('?',), 1, False, 'x', 'n'), ImplSig(('i',), 1, False, 'i', 'n')], 'i', None),
    ([ImplSig(('i',), 1, False, 'i', 'n'), ImplSig(('?',), 1, False, 'x', 'n')], 'i', 'i'),
    # An implementation the defaults let take one argument is reached only by
    # the selector's SECOND half, so the exact-arity one wins wherever there
    # is one, whatever the declaration order
    ([ImplSig(('i', 'f'), 1, False, 'x', 'ny'),
      ImplSig(('i',), 1, False, 'i', 'n')], 'i', 'i'),
    # ... and answers the pin itself where there is not
    ([ImplSig(('i', 'i'), 1, False, 'x', 'ny'),
      ImplSig(('f',), 1, False, 'f', 'n')], 'i', 'x'),
    # A default the exact pass would REJECT (an int default under a float
    # parameter) takes the implementation out of that pass entirely
    ([ImplSig(('i', 'f'), 1, False, 'x', 'nn')], 'i', None),
    # A default this pass cannot type leaves the site unanswerable
    ([ImplSig(('i', 'i'), 1, False, 'x', 'n?')], 'i', None),
    # A parameter beyond the pin with no default at all: the bind fails
    ([ImplSig(('i', 'i'), 2, False, 'x', 'nn')], 'i', None),
    # A keyword-only parameter is bound and type-checked with the rest, so a
    # fitting default keeps the implementation and a rejected one drops it
    ([ImplSig(('i',), 1, False, 'x', 'ny')], 'i', 'x'),
    ([ImplSig(('i',), 1, False, 'x', 'nn')], 'i', None),
    # A ``*args`` implementation can take any shape and matches none exactly
    ([ImplSig((), 0, True, 'x', ''), ImplSig(('i',), 1, False, 'i', 'n')], 'i', None),
    # An arity nothing answers
    ([ImplSig(('i',), 1, False, 'i', 'n')], 'ii', None),
])
def __test_the_static_selection_is_the_exact_pass__(impls: list[ImplSig], pin: str,
                                                    expected: str | None):
    """ Only what the selector's first pass settles is answered here """
    assert overload_pick(impls, pin) == expected


DEFAULTED_GROUPS = '''
from pynecore.core.overload import overload

@overload
def h(x: int, y: int = 0) -> int:
    return x / 2 + y

@overload
def h(x: float) -> float:
    return x / 2 + 100

@overload
def g(x: int) -> str:
    return 'int-impl'

@overload
def g(x: float) -> str:
    return 'float-impl'
'''


def __test_a_defaulted_parameter_keeps_the_selected_return_type__():
    """
    The exact pass binds the omitted defaults, so the pin still names one.

    ``h(1)`` reaches ``h(x: int, y: int = 0)``: the selector's positional half
    skips it on arity, and its second half binds ``y`` to the default and
    type-checks that too. Reading only the first half left ``h(1)`` UNKNOWN,
    which unpinned ``g(h(1))`` and sent it to the float implementation.
    """
    table = infer_module(ast.parse(DEFAULTED_GROUPS + '''
def main(r: int):
    return g(h(1))
'''), 'test')
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['i']
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['i=0']


def __test_a_defaulted_pin_reaches_the_int_implementation__():
    """ The defaulted group runs the way the pin says, through the pipeline """
    ns, _ = _transform(DEFAULTED_GROUPS + '''
def main(r: int):
    return h(1), g(h(1))
''', 'pin_mod_defaulted')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (0.5, 'int-impl')


#: The same shape, with the int implementation's default spelled ``None``.
#: MEASURED against ``core/overload.py::_check_type``: ``None`` is not an
#: ``int``, so the selector's exact pass drops that implementation and its
#: widening pass takes the float one -- ``h(1)`` runs ``h(x: float)``.
NONE_DEFAULT_GROUPS = '''
from pynecore.core.overload import overload

@overload
def h(x: int, y: int = None) -> int:
    return 1

@overload
def h(x: float, y: float = 0.0) -> float:
    return 2.5

@overload
def g(x: int) -> str:
    return 'int-impl'

@overload
def g(x: float) -> str:
    return 'float-impl'
'''

#: ... and with an annotation that DOES take it, which is the whole point of
#: writing ``| None``: the exact pass keeps the int implementation.
OPTIONAL_DEFAULT_GROUPS = NONE_DEFAULT_GROUPS.replace('y: int = None', 'y: int | None = None')


def __test_a_none_default_the_annotation_rejects_answers_nothing__():
    """
    ``y: int = None`` takes its implementation out of the exact pass.

    ``int`` and ``int | None`` are the same Pine type, so the type character
    cannot tell them apart -- and the runtime does, because it type-checks the
    bound default like any other value. Reading the ``None`` as typeless made
    the static pass answer ``h(1)`` with the int implementation while the
    selector ran the float one, and the pin that inconsistency handed to
    ``g(h(1))`` forced the int implementation on a float value.

    The widening pass is the one that answers, as it does at run time: it takes
    the int witness for ``h(x: float, ...)``, so ``h(1)`` is a float, and the
    float-typed argument of ``g`` takes the float implementation.
    """
    table = infer_module(ast.parse(NONE_DEFAULT_GROUPS + '''
def main(r: int):
    return g(h(1))
'''), 'test')
    assert [call.pin for call in table.calls if call.callee == 'h'] == ['i=1']
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['f']
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['f=1']


def __test_a_none_default_dispatches_the_way_the_static_pass_says__():
    """ The unpinned group falls back to the values, and they agree """
    ns, _ = _transform(NONE_DEFAULT_GROUPS + '''
def main(r: int):
    return h(1), g(h(1))
''', 'pin_mod_none_default')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (2.5, 'float-impl')


def __test_an_optional_annotation_takes_the_none_default__():
    """ ``y: int | None = None`` binds, so the int implementation is picked """
    table = infer_module(ast.parse(OPTIONAL_DEFAULT_GROUPS + '''
def main(r: int):
    return g(h(1))
'''), 'test')
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['i']
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['i=0']

    ns, _ = _transform(OPTIONAL_DEFAULT_GROUPS + '''
def main(r: int):
    return h(1), g(h(1))
''', 'pin_mod_optional_default')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (1, 'int-impl')


@pytest.mark.parametrize("annotation,expected", [
    ('int', False),
    ('float', False),
    ('str', False),
    ('int | None', True),
    ('None | int', True),
    ('Optional[int]', True),
    ('Union[int, None]', True),
    ('Union[int, str]', False),
    ('Any', True),
    ('object', True),
    ('None', True),
    ('"int | None"', True),
    ('"_ExitOrderKey"', False),
    # The absence markers the Pine type reads THROUGH are not None-takers:
    # neither an ``NA`` instance nor a ``Series`` answers ``isinstance(None, ..)``
    ('NA[int]', False),
    ('int | NA', False),
    ('Series[float]', False),
    ('list[int]', False),
])
def __test_which_annotations_take_a_none_default__(annotation: str, expected: bool):
    """ Mirrors ``_check_type(None, hint, strict=True)``, spelling by spelling """
    assert annotation_takes_none(ast.parse(annotation, mode='eval').body) is expected


def __test_a_missing_annotation_takes_a_none_default__():
    """ An unannotated parameter reads as ``Any`` at run time, which takes it """
    assert annotation_takes_none(None) is True


@pytest.mark.parametrize("declared,default,takes_none,expected", [
    ('i', NONE_DEFAULT, True, FIT_OMISSIBLE),
    ('i', NONE_DEFAULT, False, FIT_REQUIRED),
    # ``na`` and the dynamic-default sentinel are omissible whatever the
    # annotation says: the first satisfies every type, the second is skipped
    ('i', TYPELESS, False, FIT_OMISSIBLE),
    ('i', 'i', False, FIT_OMISSIBLE),
    ('f', 'i', False, FIT_REQUIRED),
    ('i', None, False, FIT_REQUIRED),
])
def __test_the_fit_of_one_default__(declared: str, default: str | None,
                                    takes_none: bool, expected: str):
    """ Only a literal ``None`` is decided by the annotation's None-acceptance """
    assert default_fit(declared, default, takes_none) == expected


def __test_a_required_parameter_beyond_the_pin_answers_nothing__():
    """
    A parameter the call does not fill and no default covers is a non-match.

    ``h(1)`` cannot bind ``h(x: int, y: int)`` at all, in either pass, so the
    selector's widening pass takes the int witness for ``h(x: float)``: the call
    is a float, the way the runtime selects it.
    """
    table = infer_module(ast.parse('''
from pynecore.core.overload import overload

@overload
def h(x: int, y: int) -> int:
    return x / 2 + y

@overload
def h(x: float) -> float:
    return x / 2

def main(r: int):
    return h(1)
'''), 'test')
    assert [call.pin for call in table.calls if call.callee == 'h'] == ['i=1']
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['f']


def __test_a_module_level_pin_binds_in_place__():
    """
    Module level owns no anchor, so a pinned site binds its dispatcher there.

    ``R / 8`` is int-TYPED with a fractional value: left alone, the top-level
    call resolves from the VALUE and runs the float implementation. Only the
    int-typed site names the int implementation, the float one the float one.
    """
    ns, dump = _transform(PIN_SRC + '''
R = 14
pinned = f(R / 8)
plain = f(R * 1.0)
''', 'pin_mod_module')
    assert "pinned = __bind_pinned·__(f, 'i=0')(R / 8)" in dump
    assert "plain = __bind_pinned·__(f, 'f=1')(R * 1.0)" in dump
    assert ns['pinned'] == 1.0
    assert ns['plain'] == 2.0


#: Two classes a group may tell apart at a position the pin has no witness for
PIVOT_TY = object_ty(class_id('test', 'Pivot'))
HOLDER_TY = object_ty(class_id('test', 'Holder'))


@pytest.mark.parametrize("impls,pin,shapes,expected", [
    # An object of a known class is taken only by the implementation declaring
    # its class: the selector checks the object against the parameter's class
    ([ImplSig((HOLDER_TY, 'i'), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 'i'), 2, False, 'B', 'nn')], '*i', [PIVOT_TY, INT], 'B'),
    # ... or by one declaring nothing there, which may take anything: an earlier
    # such one leaves the site unanswerable, a later one is never reached
    ([ImplSig(('?', 'i'), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 'i'), 2, False, 'B', 'nn')], '*i', [PIVOT_TY, INT], None),
    ([ImplSig((PIVOT_TY, 'i'), 2, False, 'B', 'nn'),
      ImplSig(('?', 'i'), 2, False, 'A', 'nn')], '*i', [PIVOT_TY, INT], 'B'),
    # A position of no known class says nothing, so the first implementation
    # that could take it leaves the site unanswerable
    ([ImplSig((HOLDER_TY, 'i'), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 'i'), 2, False, 'B', 'nn')], '*i', [OBJECT, INT], None),
    ([ImplSig((PIVOT_TY, 'i'), 2, False, 'B', 'nn')], '*i', [], None),
    # Two classed positions both have to match
    ([ImplSig((PIVOT_TY, HOLDER_TY), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, PIVOT_TY), 2, False, 'B', 'nn')], '**', [PIVOT_TY, PIVOT_TY], 'B'),
    # The second half of the pass reads a classed position the same way
    ([ImplSig((HOLDER_TY, 'i'), 1, False, 'A', 'ny'),
      ImplSig((PIVOT_TY, 'i'), 1, False, 'B', 'ny')], '*', [PIVOT_TY], 'B'),
    # A typeless na is taken for whatever a position declares -- a class, a
    # string, a container -- so the first implementation the other positions
    # let through runs; a position declaring nothing still says nothing
    ([ImplSig((HOLDER_TY, 'i'), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 'i'), 2, False, 'B', 'nn')], '*i', [TYPELESS, INT], 'A'),
    ([ImplSig((PIVOT_TY, 'i'), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 's'), 2, False, 'B', 'nn')], '**', [PIVOT_TY, TYPELESS], 'A'),
    ([ImplSig((HOLDER_TY, 'a:' + PIVOT_TY), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 's'), 2, False, 'B', 'nn')], '**', [PIVOT_TY, TYPELESS], 'B'),
    ([ImplSig((PIVOT_TY, '?'), 2, False, 'A', 'nn'),
      ImplSig((PIVOT_TY, 's'), 2, False, 'B', 'nn')], '**', [PIVOT_TY, TYPELESS], None),
])
def __test_the_static_selection_reads_an_object_by_its_class__(impls: list[ImplSig], pin: str,
                                                               shapes: list[str],
                                                               expected: str | None):
    """A position the pin has no witness for is read from the argument's class"""
    assert overload_pick(impls, pin, shapes) == expected
    choice = overload_choice(impls, pin, shapes)
    assert choice is (None if expected is None else next(i for i in impls if i.ret == expected))


@pytest.mark.parametrize("impls,chars,shapes,kw_chars,kw_shapes,expected", [
    # A keyword binds to the parameter of its name, and the first implementation
    # whose every bound value fits is the one that runs
    ([ImplSig((PIVOT_TY, 's'), 2, False, 'A', 'nn', ('this', 'sess')),
      ImplSig((PIVOT_TY, 's', 's'), 3, False, 'B', 'nnn', ('this', 'sess', 'tz'))],
     '*', [PIVOT_TY], {'sess': 's', 'tz': 's'}, {'sess': 's', 'tz': 's'}, 'B'),
    ([ImplSig((PIVOT_TY, 's'), 2, False, 'A', 'nn', ('this', 'sess')),
      ImplSig((PIVOT_TY, 's', 's'), 3, False, 'B', 'nnn', ('this', 'sess', 'tz'))],
     '*', [PIVOT_TY], {'sess': 's'}, {'sess': 's'}, 'A'),
    # The receiver's class rules an implementation out before its keywords are read
    ([ImplSig((HOLDER_TY, 's'), 2, False, 'A', 'nn', ('this', 'sess')),
      ImplSig((PIVOT_TY, 's'), 2, False, 'B', 'nn', ('this', 'sess'))],
     '*', [PIVOT_TY], {'sess': 's'}, {'sess': 's'}, 'B'),
    # A parameter left out has to have a default the exact pass accepts
    ([ImplSig((PIVOT_TY, 's', 'i'), 2, False, 'A', 'nny', ('this', 'sess', 'n'))],
     '*', [PIVOT_TY], {'sess': 's'}, {'sess': 's'}, 'A'),
    ([ImplSig((PIVOT_TY, 's', 'i'), 3, False, 'A', 'nnn', ('this', 'sess', 'n'))],
     '*', [PIVOT_TY], {'sess': 's'}, {'sess': 's'}, None),
    # A keyword no parameter is named by fails the bind, unless a keyword-only
    # parameter the shape does not name could take it
    ([ImplSig((PIVOT_TY,), 1, False, 'A', 'n', ('this',)),
      ImplSig((PIVOT_TY, 's'), 2, False, 'B', 'nn', ('this', 'sess'))],
     '*', [PIVOT_TY], {'sess': 's'}, {'sess': 's'}, 'B'),
    ([ImplSig((PIVOT_TY,), 1, False, 'A', 'ny', ('this',)),
      ImplSig((PIVOT_TY, 's'), 2, False, 'B', 'nn', ('this', 'sess'))],
     '*', [PIVOT_TY], {'sess': 's'}, {'sess': 's'}, None),
    # A keyword for a position already filled fails the bind
    ([ImplSig((PIVOT_TY, 's'), 2, False, 'A', 'nn', ('this', 'sess'))],
     '*s', [PIVOT_TY, 's'], {'sess': 's'}, {'sess': 's'}, None),
    # A keyword of no witness and no class says nothing
    ([ImplSig((PIVOT_TY, 's'), 2, False, 'A', 'nn', ('this', 'sess'))],
     '*', [PIVOT_TY], {'sess': '*'}, {'sess': OBJECT}, None),
    # A typeless na keyword is taken by whatever its parameter declares
    ([ImplSig((PIVOT_TY, 's'), 2, False, 'A', 'nn', ('this', 'sess')),
      ImplSig((PIVOT_TY, HOLDER_TY), 2, False, 'B', 'nn', ('this', 'top'))],
     '*', [PIVOT_TY], {'top': '*'}, {'top': TYPELESS}, 'B'),
])
def __test_a_call_with_keywords_is_bound_by_name__(impls: list[ImplSig], chars: str,
                                                   shapes: list[str], kw_chars: dict[str, str],
                                                   kw_shapes: dict[str, str],
                                                   expected: str | None):
    """A keyword call is the selector's bind-and-check pass, read statically"""
    assert overload_bind(impls, chars, shapes, kw_chars, kw_shapes) == expected
    choice = overload_binding(impls, chars, shapes, kw_chars, kw_shapes)
    assert choice is (None if expected is None else next(i for i in impls if i.ret == expected))


LINES_TY = 'a:' + PIVOT_TY
BOXES_TY = 'a:' + HOLDER_TY
PIVOT_MAP_TY = 'M:s:' + PIVOT_TY


@pytest.mark.parametrize("impls,pin,shapes,expected", [
    # A container of a known element: the declared type itself takes it for
    # sure, a container of another kind never does
    ([ImplSig((PIVOT_MAP_TY,), 1, False, 'M', 'n'), ImplSig((LINES_TY,), 1, False, 'A', 'n')],
     '*', [LINES_TY], [1]),
    ([ImplSig((LINES_TY,), 1, False, 'A', 'n'), ImplSig((BOXES_TY,), 1, False, 'B', 'n')],
     '*', [LINES_TY], [0]),
    # ... while one of the same kind with another element may take it: an empty
    # one passes for the first ``list[...]`` parameter
    ([ImplSig((BOXES_TY,), 1, False, 'B', 'n'), ImplSig((LINES_TY,), 1, False, 'A', 'n')],
     '*', [LINES_TY], [0, 1]),
    # A scalar or a class parameter never takes a container
    ([ImplSig(('f',), 1, False, 'F', 'n'), ImplSig((PIVOT_TY,), 1, False, 'P', 'n'),
      ImplSig((LINES_TY,), 1, False, 'A', 'n')], '*', [LINES_TY], [2]),
    # A container of numbers is matched by a sampled element that may not be of
    # its declared type, so not even the declared type is sure to take it
    ([ImplSig(('M:s:f',), 1, False, 'F', 'n'), ImplSig(('M:s:i',), 1, False, 'I', 'n')],
     '*', ['M:s:f'], [0, 1]),
    # A color: a color parameter takes it, any other parameter may take the
    # typeless na a color na is
    ([ImplSig(('c',), 1, False, 'C', 'n'), ImplSig(('s',), 1, False, 'S', 'n')],
     '*', ['c'], [0]),
    ([ImplSig(('s',), 1, False, 'S', 'n'), ImplSig(('c',), 1, False, 'C', 'n')],
     '*', ['c'], [0, 1]),
    # No implementation is sure to take the call: the one left is what runs where
    # every other one is out of the widening pass's reach too -- a class mismatch is
    ([ImplSig((PIVOT_TY, 'f'), 2, False, 'P', 'nn'), ImplSig((HOLDER_TY, 'f'), 2, False, 'H', 'nn')],
     '**', [HOLDER_TY, 'f'], [1]),
    # ... an int argument for a float parameter is not, the widening pass takes it
    ([ImplSig(('f', 's'), 2, False, 'F', 'nn'), ImplSig(('i', '?'), 2, False, 'I', 'nn')],
     'is', [], None),
    # ... a bool for an int parameter is
    ([ImplSig(('b', 's'), 2, False, 'B', 'nn'), ImplSig(('i', '?'), 2, False, 'I', 'nn')],
     'is', [], [1]),
    # ... nor is the na of a class for a float parameter, which that pass takes
    ([ImplSig(('f', 'f'), 2, False, 'F', 'nn'), ImplSig((PIVOT_TY, '?'), 2, False, 'P', 'nn')],
     '*f', [PIVOT_TY, 'f'], None),
    # ... nor a default the exact pass rejects, which that pass may take
    ([ImplSig(('i', 'f'), 1, False, 'X', 'nn'), ImplSig(('?',), 1, False, 'U', 'n')],
     'i', [], None),
    # ... while an arity the call does not meet is
    ([ImplSig(('i', 'f'), 2, False, 'X', 'nn'), ImplSig(('?',), 1, False, 'U', 'n')],
     'i', [], [1]),
])
def __test_the_candidates_are_what_the_selector_can_run__(impls: list[ImplSig], pin: str,
                                                          shapes: list[str],
                                                          expected: list[int] | None):
    """The candidate walk collects what may run, and answers only where nothing else can"""
    assert overload_candidates(impls, pin, shapes) == expected


def __test_the_candidate_readings_hold_at_run_time__():
    """
    The values behind the container, color and widening readings, through the
    runtime selector itself: an empty list passes for the first ``list[...]``
    implementation whatever its element, a list of objects for the one of its
    class, a color na (the typeless na) for a string parameter, the na of a class
    for a float parameter in the widening pass
    """
    def boxes(x: list[Box]) -> str:
        return 'boxes'

    def lines(x: list[Line]) -> str:
        return 'lines'

    def mapped(x: dict[str, Line]) -> str:
        return 'map'

    group = [Implementation(boxes), Implementation(lines)]
    assert _select(group, ([],), {}).func is boxes
    assert _select(group, ([NA(Line)],), {}).func is lines
    assert _select([Implementation(mapped), Implementation(lines)], ([],), {}).func is lines
    assert _select([Implementation(mapped), Implementation(lines)], (NA(list),), {}).func is lines

    def text(x: str) -> str:
        return 'str'

    def color(x: Color) -> str:
        return 'color'

    assert _select([Implementation(text), Implementation(color)], (NA(None),), {}).func is text
    assert _select([Implementation(text), Implementation(color)], (NA(Color),), {}).func is color

    def number(x: float) -> str:
        return 'float'

    assert _select([Implementation(number)], (NA(Line),), {}).func is number


CANDIDATE_GROUPS = '''
from pynecore.core.overload import overload
from pynecore.lib import array, na
from pynecore.types import Box, Label, Line

@overload
def count(x: list[Line]) -> int:
    return array.size(x)

@overload
def count(x: list[Box]) -> int:
    return array.size(x) + 1

@overload
def first(x: list[Box]) -> Box:
    return array.get(x, 0)

@overload
def first(x: list[Line]) -> Line:
    return array.get(x, 0)

@overload
def tag(x: dict[str, Label], y: str) -> str:
    return y

@overload
def tag(x: list[Line], y: str) -> int:
    return 1
'''


def __test_a_call_the_candidates_agree_on_is_typed__():
    """
    ``count(lines)`` runs the ``list[Line]`` implementation, declared first;
    ``first(lines)`` can run either by its values (an empty array of lines
    passes for the ``list[Box]`` one, declared first), so its pin names the
    ``list[Line]`` one Pine selects by index, and the call has its type; ``tag``
    takes an array by its kind, so a map runs the map implementation
    """
    table = infer_module(ast.parse(CANDIDATE_GROUPS + '''
def main():
    lines: list[Line] = array.new_line(0)
    labels: dict[str, Label] = {}
    n = count(lines)
    f = first(lines)
    t = tag(labels, 'x')
    u = tag(lines, 'x')
    return n
'''), 'test')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert (types['n'], types['f'], types['t'], types['u']) == ('i', 'o:lib#Line', 's', 'i')


@pytest.mark.parametrize("bare,expected", [
    # Unannotated: the selector checks the position against ``Any``
    ('11', 0),
    # A type this pass cannot read may reject the value: nothing is claimed
    ('', None),
    ('10', None),
])
def __test_an_unannotated_position_takes_every_argument__(bare: str, expected: int | None):
    """ The exact pass takes any value for a parameter without an annotation """
    impls = [ImplSig(params=('?', '?'), required=2, open_ended=False, ret='f', fits='nn',
                     names=('x', 'y'), bare=bare),
             ImplSig(params=('i',), required=1, open_ended=False, ret='i', fits='n',
                     names=('x',))]
    choice = overload_choice(impls, '*f', ['a:i', 'f'])
    assert (None if choice is None else impls.index(choice)) == expected
    # Either way it is the only implementation the call can reach
    assert overload_candidates(impls, '*f', ['a:i', 'f']) == [0]


@pytest.mark.parametrize("bare,expected", [('11', 0), ('', None)])
def __test_an_unannotated_position_takes_every_keyword_argument__(bare: str,
                                                                  expected: int | None):
    """ A keyword bound to a parameter without an annotation is taken like a positional one """
    impls = [ImplSig(params=('?', '?'), required=2, open_ended=False, ret='f', fits='nn',
                     names=('x', 'y'), bare=bare)]
    choice = overload_binding(impls, '*', ['a:i'], {'y': '*'}, {'y': 'o:m#A'})
    assert (None if choice is None else impls.index(choice)) == expected


# --- the indexed pin: Pine's selection where the values would run another ---

#: A group whose ``list[Line]`` implementation comes after another array one: an
#: EMPTY array of lines passes for the first ``list[...]`` implementation by its
#: values, while Pine selects by the static element type. The first implementation
#: takes a required keyword-only parameter, which the static selection leaves out
#: of its shapes: the index still counts it, it is the registration order.
INDEXED_GROUP = '''
from pynecore import Persistent
from pynecore.core.overload import overload
from pynecore.lib import array
from pynecore.types import Box, Line

@overload
def tally(x: list[int], *, k: int) -> int:
    return -1

@overload
def tally(x: list[Box]) -> str:
    return 'boxes'

@overload
def tally(x: list[Line]) -> int:
    seen: Persistent[int] = 0
    seen += 1
    return seen
'''


def __test_an_empty_container_is_pinned_by_index__():
    """
    The call is typed by the implementation Pine selects, and its pin names that
    implementation by its registration index, the left-out keyword-only one counted
    """
    tree = ast.parse(INDEXED_GROUP + '''
def main():
    lines: list[Line] = array.new_line(0)
    n = tally(lines)
    return n
''')
    table = infer_module(tree, 'test')
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'tally')
    assert get_pin(call) == '*=2'
    assert pin_index(get_pin(call)) == 2
    assert table.bindings['main']['n'].ty == INT
    assert [sig.slot for sig in table.groups['tally']] == [1, 2]


def __test_the_indexed_pin_runs_the_selected_implementation__():
    """
    The empty array of lines runs the ``list[Line]`` implementation, whose state
    persists across bars like any pinned anchor's
    """
    ns, dump = _transform(INDEXED_GROUP + '''
def main():
    lines: list[Line] = array.new_line(0)
    return tally(lines)
''', 'pin_mod_indexed')
    assert "'*=2'" in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert [ns['main'](state) for _ in range(3)] == [1, 2, 3]


def __test_the_indexed_pin_can_be_switched_off__(monkeypatch):
    """ ``PYNE_NO_TYPE_PIN=1`` ignores the index: the values run the first array implementation """
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    ns, _ = _transform(INDEXED_GROUP + '''
def main():
    lines: list[Line] = array.new_line(0)
    return tally(lines)
''', 'pin_mod_indexed_off')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state) == 'boxes'


def __test_a_decided_site_names_its_implementation__():
    """
    Where the selection the values make already runs Pine's implementation, the pin
    still names it by its index: an array of boxes takes the boxes implementation
    """
    tree = ast.parse(INDEXED_GROUP + '''
def main(r: int):
    boxes: list[Box] = array.new_box(0)
    s = tally(boxes)
    return s
''')
    table = infer_module(tree, 'test')
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'tally')
    assert get_pin(call) == '*=1'
    assert table.bindings['main']['s'].ty == 's'


def __test_a_float_argument_of_a_direct_call_names_the_float_implementation__():
    """
    A float-typed value the runtime may hold as a Python int runs the float
    implementation: the pin names it by its index
    """
    tree = ast.parse(PIN_SRC + '''
def main(r: float):
    return f(r)
''')
    infer_module(tree, 'test')
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'f')
    assert get_pin(call) == 'f=1'


@pytest.mark.parametrize("params,shapes,expected", [
    # Exact first, in any declaration order
    ([('f',), ('i',)], ['i'], 1),
    # An int widens to a float only where nothing takes it exactly
    ([('f',), ('s',)], ['i'], 0),
    # The element type of a container decides
    ([('a:o:lib#Box',), ('a:o:lib#Line',)], ['a:o:lib#Line'], 1),
    # No implicit conversion between container element types
    ([('a:f',)], ['a:i'], None),
    # Two exact answers are no answer
    ([('i',), ('i',)], ['i'], None),
    # An argument type that is not settled decides nothing
    ([('i',), ('f',)], ['?'], None),
    ([('i',), ('f',)], ['*'], None),
    # Nor does a declared type that is not
    ([('?',), ('f',)], ['f'], None),
])
def __test_pines_own_static_selection__(params, shapes, expected):
    """ ``pine_choice`` is Pine's rule over the static types alone """
    impls = [ImplSig(params=p, required=len(p), open_ended=False, ret='v', fits='n' * len(p),
                     names=('x',) * len(p)) for p in params]
    assert pine_choice(impls, shapes) == expected


def _indexed_group():
    """Two array implementations, built without the decorator."""
    def strings(x: list[str]) -> str:
        return 'strings'

    def ints(x: list[int]) -> str:
        return 'ints'
    return [Implementation(strings), Implementation(ints)]


def __test_the_index_selects_without_witnesses__():
    """ An empty array runs the implementation the index names, not the first of its kind """
    impls = _indexed_group()
    assert _anchored(impls, 'test.indexed')([]) == 'strings'
    bound = _anchored(impls, 'test.indexed', pin='*=1')
    assert bound([]) == 'ints'
    assert bound([1]) == 'ints'


def __test_a_wildcard_position_does_not_overrule_the_index__():
    """
    The type pass decided the index from every argument's static type, a wildcard
    position's included: no value is checked against it, on the first call or later
    """
    bound = _anchored(_indexed_group(), 'test.indexed_wildcard', pin='*=1')
    assert bound.__qualname__.endswith('dispatch_pinned')
    assert bound(['a']) == 'ints'
    assert bound([]) == 'ints'


@pytest.mark.parametrize("pin", ['*=2', '*=', '*=x', '**=1'])
def __test_an_index_the_group_cannot_run_falls_back__(pin: str):
    """ An index out of range, a malformed one, or one of another arity: the values decide """
    bound = _anchored(_indexed_group(), 'test.indexed_bad', pin=pin)
    assert bound([]) == 'strings'


def __test_the_index_is_ignored_when_pins_are_off__(monkeypatch):
    """ ``PYNE_NO_TYPE_PIN=1`` dispatches from the values, an index included """
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    bound = _anchored(_indexed_group(), 'test.indexed_off', pin='*=1')
    assert bound([]) == 'strings'


def __test_the_indexed_route_keeps_the_shape_guards__():
    """ A keyword spelling, another arity and a grown group all go to the values """
    impls = _indexed_group()
    bound = _anchored(impls, 'test.indexed_guards', pin='*=1')
    assert bound(x=[]) == 'strings'
    with pytest.raises(WrongOperandTypeError):
        bound([], [])

    def floats(x: list[float]) -> str:
        return 'floats'
    impls.insert(0, Implementation(floats))
    assert bound([]) == 'floats'



# --- the static resolution of a pin ----------------------------------------

def _group(*impls: ImplSig) -> list[ImplSig]:
    """The shapes as one whole group, each with its registration index."""
    return [impl._replace(slot=slot, size=len(impls)) for slot, impl in enumerate(impls)]


#: A ``ta.highest``-shaped group: ``(source: float, length: int)`` and ``(length: int)``
HIGHEST = (ImplSig(('f', 'i'), 2, False, 'src', 'nn'), ImplSig(('i',), 1, False, 'len', 'n'))


@pytest.mark.parametrize("impls,pin,expected", [
    # The exact pass first: the int implementation of either declaration order
    ((ImplSig(('f',), 1, False, 'f', 'n'), ImplSig(('i',), 1, False, 'i', 'n')), 'i', 'i'),
    # No exact match: the widening pass takes the int witness for the float
    ((ImplSig(('f',), 1, False, 'f', 'n'),), 'i', 'f'),
    # ... and the float witness for an int
    ((ImplSig(('i',), 1, False, 'i', 'n'),), 'f', 'i'),
    # ... but nothing else: a bool is no number to either pass
    ((ImplSig(('i',), 1, False, 'i', 'n'),), 'b', None),
    # ``ta.highest(int_source, 14)``: no implementation takes two ints exactly,
    # and the widening pass takes the int source for ``source: float``
    (HIGHEST, 'ii', 'src'),
    (HIGHEST, 'i', 'len'),
    # The wildcard rule: the witnessed positions alone, and only a single
    # implementation taking them is an answer
    (HIGHEST, '*i', 'src'),
    ((ImplSig(('f', 'i'), 2, False, 'A', 'nn'),
     ImplSig(('s', 'i'), 2, False, 'B', 'nn')), '*i', None),
    # ... in the first pass any implementation answers, the widening one too
    ((ImplSig(('f', 'f'), 2, False, 'A', 'nn'),
     ImplSig(('s', 'b'), 2, False, 'B', 'nn')), '*i', 'A'),
    # A default the exact pass rejects is checked again by the widening pass
    ((ImplSig(('i', 'f'), 1, False, 'x', 'nn', wide_fits='ny'),), 'i', 'x'),
    # ... and where the shape does not say how, there is no answer
    ((ImplSig(('i', 'f'), 1, False, 'x', 'nn'),), 'i', None),
    # An acceptance spec decides a position its Pine type does not: ``int | float``
    ((ImplSig(('f',), 1, False, 'union', 'n', accepts=('if/if',)),
     ImplSig(('i',), 1, False, 'i', 'n')), 'i', 'union'),
    ((ImplSig(('f',), 1, False, 'unknown', 'n', accepts=('?',)),), 'i', None),
    # A keyword-only parameter is what the positional half checks the second
    # argument against, and this shape does not say how
    ((ImplSig(('i',), 1, False, 'x', 'nn'), ImplSig(('i', 'i'), 2, False, 'y', 'nn')),
     'ii', None),
    # A ``**kwargs`` takes shapes this does not model
    ((ImplSig(('i',), 1, False, 'x', 'n', kwarg=True),), 'i', None),
    # An unwitnessable character
    (HIGHEST, 'ix', None),
])
def __test_the_pin_resolves_the_way_the_runtime_does__(impls: tuple[ImplSig, ...], pin: str,
                                                       expected: str | None):
    """ ``resolve_pin``: both passes, the wildcard rule, and no answer where unsure """
    resolved = resolve_pin(_group(*impls), pin)
    assert (None if resolved is None else resolved.ret) == expected


def __test_only_a_whole_group_resolves_a_pin__():
    """ Shapes that are not the whole group in its order answer nothing """
    whole = _group(*HIGHEST)
    assert resolve_pin(whole, 'ii') is whole[0]
    # A group the static selection leaves an implementation out of
    assert resolve_pin([impl._replace(size=3) for impl in whole], 'ii') is None
    # ... and shapes without their registration index
    assert resolve_pin([impl._replace(slot=-1) for impl in whole], 'ii') is None


@pytest.mark.parametrize("key,pin,expected", [
    ('ta.highest', 'ii', 0),
    ('ta.highest', '*i', 0),
    ('ta.highest', 'fi', 0),
    ('ta.highest', 'i', 1),
    ('ta.pivothigh', '*ii', 0),
    # ``timestamp``: the strings are ``str`` aliases, the components ``int | float``
    ('timestamp', 's', 0),
    ('timestamp', 'iii', 2),
    ('timestamp', 'fff', 2),
    ('timestamp', 'siii', 1),
    ('timestamp', 'siiiiii', 1),
    ('timestamp', 'ii', None),
])
def __test_a_lib_pin_resolves_from_the_measured_table__(key: str, pin: str,
                                                       expected: int | None):
    """ The registry's measurements resolve a lib group's pin """
    resolved = resolve_pin(_lib_impl_sigs(lib_types()[key]), pin)
    assert (None if resolved is None else resolved.slot) == expected


#: Annotation spellings, and the names they need to be evaluated
ACCEPTS_NAMESPACE = {'Any': Any, 'Optional': Optional, 'Matrix': Matrix, 'StrLiteral': StrLiteral,
                     'PyneInt': PyneInt, 'PyneFloat': PyneFloat, 'PyneBool': PyneBool,
                     'PyneStr': PyneStr, 'Color': Color, 'NA': NA, 'Line': Line}


@pytest.mark.parametrize("annotation", [
    'int', 'float', 'bool', 'str', 'int | float', 'float | int', 'str | None', 'Optional[float]',
    'PyneInt', 'PyneFloat', 'PyneBool', 'PyneStr', 'Any', 'object', 'StrLiteral', 'Color',
    'NA', 'NA[int]', 'int | NA', 'Line', 'list', 'list[int]', 'list[float]', 'list[PyneFloat]',
    'list[int | float]', 'Matrix', 'Matrix[int]', 'Matrix[float]', 'Matrix[str]', 'dict[str, int]',
    'tuple[int, int]',
])
def __test_an_annotation_accepts_what_the_selector_checks__(annotation: str):
    """ ``annotation_accepts`` is ``_check_type`` on every witness, in both passes """
    hint = eval(annotation, dict(ACCEPTS_NAMESPACE))  # noqa: S307
    measured = '/'.join(''.join(char for char, witness in _PIN_WITNESSES.items()
                                if _check_type(witness, hint, strict))
                        for strict in (True, False))
    assert ''.join(_PIN_WITNESSES) == PIN_WITNESSED
    classes = {'Line': class_id('lib', 'Line')}
    assert annotation_accepts(ast.parse(annotation, mode='eval').body, classes) == measured


def __test_an_ordinary_pin_names_its_implementation__():
    """ A pin the witnesses resolve carries the index, and the call is typed by it """
    source = PIN_SRC + '''
@overload
def k(x: float, y: float) -> str:
    return 'float-impl'

@overload
def k(x: str, y: int) -> int:
    return 1

def main(r: int):
    a = f(r)
    b = k(r, r)
    return a, b
'''
    table = infer_module(ast.parse(source), 'test')
    assert [call.pin for call in table.calls if call.callee == 'f'] == ['i=0']
    # ``k(r, r)``: only the widening pass takes two ints, as the float implementation
    assert [call.pin for call in table.calls if call.callee == 'k'] == ['ii=0']
    assert table.bindings['main']['b'].ty == 's'
    ns, dump = _transform(source, 'pin_mod_ordinary')
    assert "'ii=0'" in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3)[1] == 'float-impl'


def __test_a_lib_pin_names_its_implementation__():
    """ ``timestamp`` and ``ta.highest`` sites carry the index the registry resolves """
    _, calls = _call_site('''
from pynecore import lib
R = lib.input.int(14)
stamp = lib.timestamp(2020, 1, R)
top = lib.ta.highest(lib.bar_index, R)
''', 'lib.timestamp')
    pins = {call.callee: call.pin for call in calls}
    assert pins['timestamp'] == 'iii=2'
    assert pins['ta.highest'] == 'ii=0'
    assert {call.callee: call.ty for call in calls}['ta.highest'] == 'f'


@pytest.mark.parametrize("pin", ['iii=2', 'siii=1', 'fff=2'])
def __test_an_indexed_pin_keeps_the_stateless_fast_path__(pin: str):
    """ A fully witnessed indexed pin binds through ``__pyne_pinned_bind__`` """
    bound = lib.timestamp.__pyne_bind__(pin)
    assert bound.__qualname__.endswith('_bind_timestamp_components.<locals>.cached')


def __test_an_indexed_wildcard_pin_takes_the_fast_path__():
    """ A wildcard position is checked by nothing, so it takes the fast path too """
    bound = lib.timestamp.__pyne_bind__('*iii=1')
    assert bound.__qualname__.endswith('_bind_timestamp_components.<locals>.cached')


def _numbers_group() -> list[Implementation]:
    """An int and a float implementation, built without the decorator."""
    def ints(x: int) -> str:
        return 'int'

    def floats(x: float) -> str:
        return 'float'
    return [Implementation(ints), Implementation(floats)]


def __test_the_index_is_what_runs__():
    """ The index is the type pass's selection, and the runtime does not select again """
    bound = _anchored(_numbers_group(), 'test.contradicted', pin='i=1')
    assert bound(1) == 'float'


def __test_an_index_whose_witness_the_implementation_rejects_falls_back__():
    """ A string implementation takes no int witness: the index is not this group's """
    def strings(x: str) -> str:
        return 'str'
    impls = _numbers_group() + [Implementation(strings)]
    assert _anchored(impls, 'test.foreign_index', pin='i=2')(1) == 'int'


@pytest.mark.parametrize("pin", ['i', 'f', '*'])
def __test_a_pin_without_an_index_dispatches_from_the_values__(pin: str):
    """ No index: the type pass named no implementation, so the values select """
    bound = _anchored(_numbers_group(), 'test.unindexed', pin=pin)
    assert bound.__qualname__.endswith('.dispatch')
    assert bound(1) == 'int' and bound(1.5) == 'float'


# --- the static decision ---------------------------------------------------

#: A group whose second position is a class the pin has no witness for:
#: ``(int, A)`` and ``(str, B)``
CLASS_POSITION_GROUP = '''
from pynecore.core.overload import overload
from pynecore.core.pine_udt import udt

@udt
class A:
    v: int = 0

@udt
class B:
    v: int = 0

def make(which: int):
    return B() if which else A()

@overload
def kind(n: int, a: A) -> str:
    return 'int-a'

@overload
def kind(s: str, b: B) -> str:
    return 'str-b'
'''


def __test_a_call_without_an_int_argument_names_its_implementation__():
    """ A string argument decides the group as an int one does: the pin names it """
    node, _ = _call_site('''
from pynecore import lib
result = lib.timestamp("2024-01-01 00:00")
''', 'lib.timestamp')
    assert get_pin(node) == 's=0'


def __test_an_argument_of_unknown_type_decides_nothing__():
    """
    A value that is an ``A`` or a ``B`` has no known type: the call carries no
    pin, even where its int argument alone would name one implementation, and is
    reported
    """
    tree = ast.parse(CLASS_POSITION_GROUP + '''
def main(which: int):
    return kind(1, make(which))
''')
    table = infer_module(tree, 'test')
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'kind')
    assert get_pin(call) is None
    found = [(diag.code, diag.args) for diag in table.diags if diag.code in ('E0071', 'E0072')]
    assert found == [('E0071', ('2', 'kind'))]


def __test_a_known_class_at_a_wildcard_position_is_read_by_its_type__():
    """ A udt argument is known: the pin's wildcard is read by its class and indexed """
    tree = ast.parse(CLASS_POSITION_GROUP + '''
def main():
    return kind(1, A()), kind("s", B())
''')
    table = infer_module(tree, 'test')
    pins = [get_pin(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'kind']
    assert pins == ['i*=0', 's*=1']
    assert not [diag for diag in table.diags if diag.code in ('E0071', 'E0072')]


def __test_an_undecided_call_no_implementation_takes_stops_the_run__():
    """
    ``kind(1, B())`` through an argument of unknown type: no implementation takes
    an int with a ``B``, and the call, carrying no index, is rejected by the values
    instead of running the int implementation
    """
    ns, dump = _transform(CLASS_POSITION_GROUP + '''
def main(which: int):
    return kind(1, make(which))
''', 'pin_mod_undecided')
    assert "'i*=0'" not in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with pytest.raises(WrongOperandTypeError):
        ns['main'](state, 1)


# --- keyword calls -------------------------------------------------------


def __test_a_keyword_call_is_pinned_by_its_types__():
    """
    ``x=r / 8`` is int-typed with a fractional value: the index names the int
    implementation, which the values alone would never run
    """
    ns, dump = _transform(PIN_SRC + '''
def main(r: int):
    return f(x=r / 8), f(x=r * 1.0)
''', 'pin_mod_keyword')
    assert "__bind_any·__(__state__, 0, f, '=0')" in dump
    assert "__bind_any·__(__state__, 1, f, '=1')" in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == (1.0, 2.0)


def __test_a_keyword_after_positions_is_pinned_with_them__():
    """ The pin's characters are the positional ones; the index covers the keyword """
    ns, dump = _transform('''
from pynecore.core.overload import overload

@overload
def g(a: int, b: int = 0) -> str:
    return 'int'

@overload
def g(a: float, b: float = 0.0) -> str:
    return 'float'

def main(r: int):
    return g(r / 8, b=r / 8), g(r / 8, b=r * 1.0)
''', 'pin_mod_keyword_mixed')
    assert "__bind_any·__(__state__, 0, g, 'i=0')" in dump
    # An int position with a float keyword: Pine widens the int to the float one
    assert "__bind_any·__(__state__, 1, g, 'i=1')" in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == ('int', 'float')


def __test_a_pine_spelled_keyword_binds_its_renamed_parameter__():
    """ ``len=`` reaches ``len__ren__`` under the pin, as it does under the values """
    ns, dump = _transform('''
from pynecore.core.overload import overload

@overload
def h(len__ren__: int) -> str:
    return 'int'

@overload
def h(len__ren__: float) -> str:
    return 'float'

def main(r: int):
    return h(len=r / 8)
''', 'pin_mod_keyword_renamed')
    assert "__bind_any·__(__state__, 0, h, '=0')" in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == 'int'


def __test_a_keyword_of_unknown_type_is_reported_and_not_pinned__():
    """ A keyword argument of unknown type decides nothing, and is named in the report """
    tree = ast.parse(CLASS_POSITION_GROUP + '''
def main(which: int):
    return kind(1, a=make(which))
''')
    table = infer_module(tree, 'test')
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'kind')
    assert get_pin(call) is None
    found = [(diag.code, diag.args) for diag in table.diags if diag.code in ('E0071', 'E0072')]
    assert found == [('E0071', ('`a`', 'kind'))]


def __test_a_lib_keyword_call_is_pinned__():
    """ A registry group called by keyword names its implementation by its index """
    node, _ = _call_site('''
from pynecore import lib
R = lib.input.int(14)
result = lib.ta.lowest(source=lib.low, length=R)
''', 'lib.ta.lowest')
    assert get_pin(node) == '=0'


def __test_the_pinned_route_binds_keywords_and_falls_back_on_another_shape__():
    """
    A keyword call runs the indexed implementation with no value check; a
    keyword the implementation does not take, and another positional count, are
    not the shape the pin was computed for and go to the values
    """
    def narrow(x: int, y: int = 0) -> str:
        return f'int:{x}:{y}'

    def wide(x: float, z: float = 0.0) -> str:
        return f'float:{x}:{z}'
    impls = [Implementation(narrow), Implementation(wide)]
    bound = _anchored(impls, 'test.keyword', pin='=0')
    assert bound(x=1.75) == 'int:1.75:0'
    assert bound(x=1.75, y=2) == 'int:1.75:2'
    assert bound(x=1.75, z=2.0) == 'float:1.75:2.0'
    assert bound(1.75) == 'float:1.75:0.0'
    with pytest.raises(WrongOperandTypeError):
        bound(x='s', z='s')


__test_helper_TWIN_SETUP = '''
from pynecore.core.overload import overload
from pynecore.lib import close, script
from pynecore.types import Simple


@script.indicator("twins")
def main():
    @overload
    def kw_only(x: float, *, scale: float = 1.0):
        return 1.0

    @overload
    def kw_only(x: float, *, scale: int = 1):
        return 2.0

    @overload
    def twin(x: float):
        return 1.0

    @overload
    def twin(x: float):
        return 2.0

    @overload
    def renamed(x: float):
        return 1.0

    @overload
    def renamed(y: float):
        return 2.0

    @overload
    def marked(x: Simple[float]):
        return 1.0

    @overload
    def marked(x: float):
        return 2.0

    a = kw_only(close)
    b = twin(close)
    c = marked(close)
    d = renamed(close)
'''


def __test_an_indistinguishable_choice_is_undecided__():
    """Twins the static types cannot tell apart -- identical, renamed or keyword-only only --
    get no index and E0072; marked ones are indexed"""
    tree = ImportNormalizerTransformer().visit(ast.parse(__test_helper_TWIN_SETUP))
    table = infer_module(tree, 'test')
    pins = {ast.unparse(node.func): get_pin(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) in ('kw_only', 'twin', 'renamed', 'marked')}
    assert pins['kw_only'] is None
    assert pins['twin'] is None
    assert pins['renamed'] is None
    assert pins['marked'] == 'f=1'
    undecided = sorted(diag.args[0] for diag in table.diags if diag.code == 'E0072')
    assert undecided == ['kw_only', 'renamed', 'twin'], [(d.code, d.args) for d in table.diags]
