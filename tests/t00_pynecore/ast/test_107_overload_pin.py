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

from pynecore.core.instance_state import _make_state
from pynecore.core.overload import Implementation, _anchored  # noqa: internal API
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    FIT_OMISSIBLE, FIT_REQUIRED, ImplSig, NONE_DEFAULT, TYPELESS, annotation_takes_none,
    default_fit, get_pin, get_ty, overload_pick,
)
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout

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
    """ The argument carries TradingView's type, and an int one is pinned """
    node, calls = _call_site(MEASURED_SETUP.format(expression=expression))
    assert get_ty(node.args[0]) == expected, expression
    pin = [call.pin for call in calls if call.callee == 'f'][-1]
    assert pin == ('i' if expected == 'i' else None), expression
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
    assert get_pin(node) == 'i'


@pytest.mark.parametrize("call", [
    'f(R * 1.0)',            # nothing int-typed: the values already agree
    'f(x=R / 8)',            # a keyword spelling is not a position
    'f(*args)',              # an unpacking hides which position is which
    'f(array.new_float(1))',  # an object: no single value witnesses it
    'f(unknown)',            # untypable, so there is nothing to pin on
])
def __test_the_unpinnable_shapes__(call: str):
    """ A shape the pin cannot describe is declined, never guessed """
    node, _ = _call_site(MEASURED_SETUP.replace(
        'import bar_index', 'import array, bar_index').format(expression='R').replace(
        'result = f(R)',
        'args = [1]\n'
        '    unknown = array.first(array.new_float(1))\n'
        f'    result = {call}'))
    assert get_pin(node) is None


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
    """ The pinned site carries the type characters, the others are unchanged """
    _, dump = _transform(PIN_SRC + '''
def main(r: int):
    return f(r / 8), f(r * 1.0)
''', 'pin_mod_a')
    assert "__bind_any·__(__state__, 0, f, 'i')" in dump
    assert '__bind_any·__(__state__, 1, f)' in dump


def __test_the_loop_and_comprehension_forms_carry_it_too__():
    """ The helper-only shapes take the pin in the same trailing position """
    _, dump = _transform(PIN_SRC + '''
def main(r: int):
    for _ in range(2):
        f(r / 8)
    return [f(r / 8) for _ in [0]]
''', 'pin_mod_b')
    assert "__bind_loop·__(__state__, 0, f, 'i')" in dump
    assert "__bind_loop·__(__state__, 1, f, 'i')" in dump

    # A comprehension ITERABLE forbids the walrus guard, so the whole site
    # folds into the slot helper -- which takes the pin in that position too
    _, dump = _transform(PIN_SRC.replace('-> float', '-> list')
                         .replace('return 1.0', 'return [1.0]')
                         .replace('return 2.0', 'return [2.0]') + '''
def main(r: int):
    return [v for v in f(r / 8)]
''', 'pin_mod_b2')
    assert "__bind_slot·__(__state__, 0, f, 'i')" in dump


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
    assert [call.pin for call in table.calls if call.callee == 'h'] == ['i']
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['i']
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['ii']

    ns, _ = _transform(source, 'pin_mod_join')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == 'A'


def __test_a_later_pass_erases_the_pin_it_no_longer_stands_behind__():
    """
    The loop fixpoint walks a body twice; the second walk decides.

    ``total`` starts int and widens to float over the loop, so the first walk
    pins the call and the second must not: a stamp left over from the first
    would emit a pin the inference itself rejected, and the site would keep
    calling the int implementation for the rest of the run.
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
    assert [call.pin for call in table.calls if call.callee == 'g'][-1] is None
    call = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'g'][0]
    assert get_pin(call) is None

    ns, _ = _transform(source, 'pin_mod_loop')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 3) == ['int-impl', 'float-impl', 'float-impl']


def _witness_group():
    """A two-implementation group built without the decorator, for the guards."""
    def wide(x: float) -> str:
        return 'float'
    return [Implementation(wide)]


def __test_the_pinned_route_falls_back_to_the_values__():
    """ A shape the pin was not computed for decides the old way """
    impls = _witness_group()
    bound = _anchored(impls, 'test.guard', pin='i')
    # The only implementation takes the int witness by widening
    assert bound(1) == 'float'
    # A keyword spelling and a different arity are not that shape
    assert bound(x=1) == 'float'
    with pytest.raises(TypeError, match='No matching implementation'):
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
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['i']


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
    # No exact match: the widening pass is the runtime's business, not this one
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
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['i']


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
    """
    table = infer_module(ast.parse(NONE_DEFAULT_GROUPS + '''
def main(r: int):
    return g(h(1))
'''), 'test')
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['?']
    assert [call.pin for call in table.calls if call.callee == 'g'] == [None]


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
    assert [call.pin for call in table.calls if call.callee == 'g'] == ['i']

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

    ``h(1)`` cannot bind ``h(x: int, y: int)`` at all, so the group's own type
    is what is left -- and its implementations disagree, so that is UNKNOWN.
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
    assert [call.ty for call in table.calls if call.callee == 'h'] == ['?']


def __test_a_module_level_pin_binds_in_place__():
    """
    Module level owns no anchor, so a pinned site binds its dispatcher there.

    ``R / 8`` is int-TYPED with a fractional value: left alone, the top-level
    call resolves from the VALUE and runs the float implementation. Only the
    pinned site is rewritten -- the float one keeps the plain call.
    """
    ns, dump = _transform(PIN_SRC + '''
R = 14
pinned = f(R / 8)
plain = f(R * 1.0)
''', 'pin_mod_module')
    assert "pinned = __bind_pinned·__(f, 'i')(R / 8)" in dump
    assert 'plain = f(R * 1.0)' in dump
    assert ns['pinned'] == 1.0
    assert ns['plain'] == 2.0
