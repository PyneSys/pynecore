"""
Pine's types are SHAPED: a value knows its class, a container its element.

``int`` and ``float`` fit in one character, but the types a Pine script
actually writes down do not. An object knows the class it is an instance of --
that is what makes ``p.price`` a float rather than an unknown. An
``array<int>`` is a different type from an ``array<float>``, which is what
makes ``array.get(a, 0)`` an int, and a ``map<string, array<float>>`` holds
float arrays all the way down. Collapsing all of that onto ``'o'`` lost the
element type at the first read, and every expression built on that read lost
its overload pin with it.

So the representation stays a string and grows a grammar::

    ty := <char> | 'o:' <class-id> | 'a:' <ty> | 'm:' <ty> | 'M:' <key> ':' <ty>

with a class id of ``<module-key>#<ClassName>``, because class identity is
(module, name) -- two modules' same-named classes are two different types, and
the shapes nest without a depth limit.

Everything that reads the LATTICE keeps reading one character: ``head()`` is
what the arithmetic, the pin and the overload selection see, so the pin wire
format never learns about shapes. And two branches producing different shapes
is a Pine compile ERROR, so it is UNKNOWN with a diagnostic that names both
types in Pine's own spelling -- never a silent widening to "some object",
which would throw away exactly the element type the next read needs.
"""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

from pynecore.core.import_hook import PIPELINE_DIGEST, analyse_source
from pynecore.core.instance_state import _make_state
from pynecore.transformers import pine_type_artifact
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.pine_type_artifact import (
    build_interface, registered, table_json, _interface_from_json,
)
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    BOOL, FLOAT, INT, OBJECT, PIN_ANY, PINE_LOOP, STR, TYPELESS, UNKNOWN, VOID,
    annotation_type, array_of, builtin_class_id, class_id, class_of, element_of, get_pin,
    get_ty, head, is_int_typed, is_shaped, join, key_of, map_of, matrix_of, object_ty, pin_for,
    render_ty, shape_mismatch, tuple_of, value_of,
)
from pynecore.transformers.pine_type_table import PineTypeTable
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide interface registry from leaking between tests."""
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()
    yield
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()


@pytest.fixture(autouse=True)
def _clean_modules():
    """Drop the modules a test imported, so a later one starts from the source."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith('shp_'):
            del sys.modules[name]


def _infer(source: str) -> tuple[ast.Module, PineTypeTable]:
    """Infer a snippet the way the pipeline does, import normalization first.

    The normalizer is what turns ``array.get`` into the ``lib.array.get``
    spelling the registry is keyed by, so without it nothing here is typed at
    all.

    :param source: Module source
    :return: (the normalized tree, its table)
    """
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    return tree, infer_module(tree, 'test')


def _types(source: str, scope: str = '') -> dict[str, str]:
    """The bindings of one scope, as name -> type."""
    _, table = _infer(source)
    return {name: binding.ty for name, binding in table.bindings.get(scope, {}).items()}


def _annotation(spelling: str, classes: dict[str, str] | None = None) -> str:
    """The type one annotation spells, resolved against a class map."""
    return annotation_type(ast.parse(spelling, mode='eval').body, classes or {})


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write a module under ``tmp_path`` and hand back its path."""
    path = tmp_path / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    return path


def _analysed(path: Path) -> tuple[ast.Module, PineTypeTable]:
    """Run the analysing half of the pipeline, cross-module resolution included."""
    analysed = analyse_source(str(path))
    assert analysed is not None, 'the module was not recognized as Pyne code'
    return analysed[0], analysed[1]


#: What the shaped tests declare, in the shape a COMPILED script has: a
#: ``@udt`` class at module level and its methods as decorated free functions.
UDT_SETUP = '''
from pynecore.core.pine_method import method, method_call
from pynecore.core.pine_udt import udt
from pynecore import lib
from pynecore.lib import array, box, map, matrix, na
from pynecore.types import NA


@udt
class Pivot:
    price: float = 0.0
    idx: int = 0
    tag: str = ""


@udt
class Holder:
    top: Pivot = NA(Pivot)
    points: list[float] = []
'''

#: The class ids the setup declares, for a module analysed under ``'test'``.
PIVOT = object_ty(class_id('test', 'Pivot'))
HOLDER = object_ty(class_id('test', 'Holder'))


# --- 1. the grammar --------------------------------------------------------

@pytest.mark.parametrize("ty,rendered", [
    (INT, 'int'),
    (FLOAT, 'float'),
    (OBJECT, 'object'),
    (PIVOT, 'Pivot'),
    (object_ty(builtin_class_id('Line')), 'Line'),
    (array_of(INT), 'array<int>'),
    (matrix_of(FLOAT), 'matrix<float>'),
    (map_of(STR, FLOAT), 'map<string, float>'),
    (map_of(STR, array_of(FLOAT)), 'map<string, array<float>>'),
    (array_of(array_of(PIVOT)), 'array<array<Pivot>>'),
])
def __test_a_shape_renders_in_pine_spelling__(ty: str, rendered: str):
    """A message names a type the way the language does, not by its character"""
    assert render_ty(ty) == rendered


def __test_a_shape_round_trips_through_its_accessors__():
    """What a constructor builds is what the reader takes back out"""
    assert element_of(array_of(INT)) == INT
    assert element_of(matrix_of(FLOAT)) == FLOAT
    assert key_of(map_of(STR, FLOAT)) == STR
    assert value_of(map_of(STR, FLOAT)) == FLOAT
    assert class_of(PIVOT) == class_id('test', 'Pivot')

    # Nesting is unbounded: the grammar reads left to right, so a map's TAIL
    # is a whole type of its own
    nested = map_of(STR, array_of(matrix_of(INT)))
    assert value_of(nested) == array_of(matrix_of(INT))
    assert element_of(element_of(value_of(nested))) == INT


def __test_an_accessor_of_the_wrong_shape_says_so__():
    """Asking an int for its element type is a question with no answer"""
    assert element_of(INT) == UNKNOWN
    assert element_of(PIVOT) == UNKNOWN
    assert key_of(array_of(INT)) == UNKNOWN
    assert value_of(OBJECT) == UNKNOWN
    assert class_of(array_of(INT)) is None
    assert class_of(OBJECT) is None


def __test_an_unknown_element_collapses_to_a_bare_object__():
    """
    A shape carrying nothing is worse than no shape at all.

    ``array_of(UNKNOWN)`` would claim a shape while saying nothing, and two
    such arrays would then read as a shape MISMATCH -- a compile error the
    program does not have -- rather than as the one honest thing they are:
    containers nothing is known about.
    """
    assert array_of(UNKNOWN) == OBJECT
    assert matrix_of(UNKNOWN) == OBJECT
    assert array_of(VOID) == OBJECT
    assert map_of(STR, UNKNOWN) == OBJECT
    # A map key is a Pine SCALAR, so a shaped key is not a map at all
    assert map_of(array_of(INT), FLOAT) == OBJECT
    assert map_of(UNKNOWN, FLOAT) == OBJECT
    assert not is_shaped(OBJECT)
    assert is_shaped(array_of(INT)) and is_shaped(PIVOT)


# --- 2. what reads a shape as one character --------------------------------

@pytest.mark.parametrize("ty", [PIVOT, array_of(INT), matrix_of(FLOAT), map_of(STR, FLOAT)])
def __test_every_shape_heads_to_an_object__(ty: str):
    """The lattice is unchanged: a shape IS an object wherever a char is read"""
    assert head(ty) == OBJECT


def __test_the_pin_never_sees_a_shape__():
    """``pin_for(['a:i', 'i'])`` and ``pin_for(['o', 'i'])`` are one question"""
    assert pin_for([array_of(INT), INT]) == pin_for([OBJECT, INT])
    # A shape has no witness value, so it takes the wildcard: the position is
    # left out of the selection, not allowed to block the int next to it
    assert pin_for([array_of(INT), INT]) == PIN_ANY + 'i'
    # ... and the arguments a shape READS OUT are pinnable as usual
    assert pin_for([INT, INT]) == 'ii'
    # A site with nothing int-typed in it is still not worth a pin
    assert pin_for([array_of(INT), FLOAT]) is None


@pytest.mark.parametrize("left,right,expected", [
    # Same shape: nothing to widen
    (array_of(INT), array_of(INT), array_of(INT)),
    (PIVOT, PIVOT, PIVOT),
    (map_of(STR, FLOAT), map_of(STR, FLOAT), map_of(STR, FLOAT)),
    # Different shapes: a Pine compile error, so UNKNOWN -- widening to an
    # object would throw away the element type the next read needs
    (array_of(INT), array_of(FLOAT), UNKNOWN),
    (array_of(INT), matrix_of(INT), UNKNOWN),
    (PIVOT, HOLDER, UNKNOWN),
    (PIVOT, array_of(INT), UNKNOWN),
    # A bare object is a shape whose class was LOST, which is a difference too
    (PIVOT, OBJECT, UNKNOWN),
    # The scalar lattice is untouched by any of it
    (INT, FLOAT, FLOAT),
    (INT, INT, INT),
    (INT, BOOL, UNKNOWN),
    # A typeless side takes the other's type, shape included
    (TYPELESS, array_of(INT), array_of(INT)),
    (PIVOT, TYPELESS, PIVOT),
])
def __test_the_join_matrix__(left: str, right: str, expected: str):
    """One expression with two types is the join of them"""
    assert join(left, right) == expected
    assert join(right, left) == expected


@pytest.mark.parametrize("left,right,conflict", [
    (array_of(INT), array_of(FLOAT), True),
    (PIVOT, HOLDER, True),
    (PIVOT, OBJECT, True),
    (array_of(INT), array_of(INT), False),
    # Not a shape conflict: an int and a float are a WIDENING, and an unknown
    # side is a type that was never established
    (INT, FLOAT, False),
    (array_of(INT), UNKNOWN, False),
])
def __test_a_shape_conflict_is_told_from_a_widening__(left: str, right: str, conflict: bool):
    """Only two objects that disagree are the compile error worth reporting"""
    assert shape_mismatch(left, right) is conflict


# --- 2b. the untyped na ----------------------------------------------------

#: A script whose block result is seeded with a bare ``na``, which is the
#: shape every compiled ``if`` without an else arrives in.
NA_SETUP = '''
from pynecore.lib import array, line, na, ta
'''


@pytest.mark.parametrize("expression,expected", [
    # A ternary with an na branch IS the other branch
    ('line.new(1, 2.0, 3, 4.0) if flag else na', object_ty(builtin_class_id('Line'))),
    ('na if flag else line.new(1, 2.0, 3, 4.0)', object_ty(builtin_class_id('Line'))),
    ('array.new_int(2, 0) if flag else na', array_of(INT)),
    ('ta.sma(src, 3) if flag else na', FLOAT),
    # MEASURED (FX:EURUSD@60, ``na_probe1``/``3``/``4``): an na operand takes
    # the other operand's type, two of them are int, and a sign keeps it
    # typeless (``int e = -na`` compiles)
    ('na + 1', INT),
    ('na + src', FLOAT),
    ('na + "x"', STR),
    ('na + na', INT),
    ('-na', TYPELESS),
    ('na', TYPELESS),
])
def __test_an_untyped_na_takes_the_type_it_meets__(expression: str, expected: str):
    """Pine's ``na`` carries no type of its own, so it can never disagree"""
    types = _types(NA_SETUP + f'''
def main(flag: bool, src: float):
    value = {expression}
    return value
''', 'main')
    assert types['value'] == expected, expression


def __test_a_block_result_seeded_with_na_takes_the_branch_type__():
    """
    The shape every compiled ``if`` without an else arrives in.

    PyneComp seeds the block temporary with a bare ``na`` and assigns the
    real value inside the branch. Reading the seed as a type of its own made
    the join UNKNOWN and took the whole expression down with it -- while Pine
    simply types the variable by what the branch stores.
    """
    types = _types(NA_SETUP + '''
def main(flag: bool, src: float):
    __block_result__ = na
    if flag:
        __block_result__ = ta.sma(src, 3)
    return __block_result__
''', 'main')
    assert types['__block_result__'] == FLOAT


def __test_a_binding_that_only_ever_holds_na_stays_typeless__():
    """
    Nothing to infer, and nothing to invent.

    Pine rejects ``x = na`` outright -- a variable needs a declared type when
    an untyped na is all it gets -- so the honest answer is the typeless one,
    not a guessed float and not an UNKNOWN that would read as a failure.
    """
    types = _types(NA_SETUP + '''
def main(flag: bool):
    empty = na
    if flag:
        empty = na
    return empty
''', 'main')
    assert types['empty'] == TYPELESS


def __test_a_typeless_argument_pins_as_a_wildcard__():
    """
    A pin is a witness VALUE per argument, and na is no witness.

    The same rule the selector already follows for a typeless default: it
    answers every annotation, so it discriminates nothing -- which is exactly
    what the wildcard says. It never makes a site pinnable on its own: without
    an int-typed argument there is nothing for the pin to correct.
    """
    assert pin_for([TYPELESS, INT]) == PIN_ANY + 'i'
    assert pin_for([TYPELESS]) is None
    assert not is_int_typed(TYPELESS)


def __test_a_typeless_type_renders_as_na__():
    """It is what a user reads in a message, so it is spelled Pine's way"""
    assert render_ty(TYPELESS) == 'na'


# --- 3. annotations --------------------------------------------------------

@pytest.mark.parametrize("spelling,expected", [
    ('list[int]', array_of(INT)),
    ('list[float]', array_of(FLOAT)),
    ('Array[int]', array_of(INT)),
    ('SequenceView[float]', array_of(FLOAT)),
    ('Matrix[float]', matrix_of(FLOAT)),
    ('dict[str, float]', map_of(STR, FLOAT)),
    ('Map[str, int]', map_of(STR, INT)),
    # Nested, which is the form a ``map<string, array<float>>`` arrives in
    ('dict[str, list[float]]', map_of(STR, array_of(FLOAT))),
    ('list[list[int]]', array_of(array_of(INT))),
    # The Pine wrappers are transparent to the shape
    ('Series[list[int]]', array_of(INT)),
    ('Persistent[dict[str, float]]', map_of(STR, FLOAT)),
    ('list[int] | NA', array_of(INT)),
    # A bare container says nothing about its elements
    ('list', OBJECT),
    ('dict', OBJECT),
    # A Pine tuple is a shape of its own -- see test_115 for the whole family
    ('tuple[int, int]', tuple_of([INT, INT])),
    # What Pine has no container for stays an opaque object
    ('set[int]', OBJECT),
    # A class the LIB publishes is nameable without importing anything
    ('Line', object_ty(builtin_class_id('Line'))),
    ('list[Label]', array_of(object_ty(builtin_class_id('Label')))),
])
def __test_an_annotation_spells_a_whole_shape__(spelling: str, expected: str):
    """The declared type is the type, containers and classes included"""
    assert _annotation(spelling) == expected


@pytest.mark.parametrize("spelling,expected", [
    ('Pivot', 'o:m#Pivot'),
    ('list[Pivot]', 'a:o:m#Pivot'),
    ('Series[Pivot]', 'o:m#Pivot'),
    ('dict[str, Pivot]', 'M:s:o:m#Pivot'),
    ('matrix[Pivot]', UNKNOWN),
])
def __test_a_class_annotation_carries_the_class_id__(spelling: str, expected: str):
    """A class is (module, name), so the module key travels with it"""
    assert _annotation(spelling, {'Pivot': 'm#Pivot'}) == expected


# --- 4. objects: fields and methods ----------------------------------------

def __test_a_field_read_has_the_declared_type__():
    """The whole point of a class id: ``p.price`` is a float, not an unknown"""
    types = _types(UDT_SETUP + '''
p = Pivot(1.0, 2, "x")
price = p.price
idx = p.idx
tag = p.tag
''')
    assert types['p'] == PIVOT
    assert (types['price'], types['idx'], types['tag']) == (FLOAT, INT, STR)


def __test_a_field_of_a_field_is_typed_too__():
    """Nesting is not a special case: the field's own type carries its class"""
    types = _types(UDT_SETUP + '''
h = Holder(NA(Pivot), array.new_float(2, 0.0))
inner = h.top
deep = h.top.price
points = h.points
first = array.get(h.points, 0)
''')
    assert types['inner'] == PIVOT
    assert types['deep'] == FLOAT
    assert types['points'] == array_of(FLOAT)
    assert types['first'] == FLOAT


def __test_a_field_the_class_does_not_have_is_reported__():
    """A misspelled field is a fact about the CLASS, so the message names it"""
    _, table = _infer(UDT_SETUP + '''
p = Pivot(1.0, 2, "x")
nope = p.nowhere
''')
    reasons = {diag.origin.reason for diag in table.diags if diag.origin is not None}
    assert 'unknown-field' in reasons
    assert [diag.render() for diag in table.diags] == [
        "'Pivot' has no field 'nowhere' "
        "-- declare 'nowhere' on 'Pivot', or read a field it has"]


def __test_a_receiver_whose_class_was_lost_points_at_the_receiver__():
    """
    The fix goes where the class went missing, not at the field.

    ``map.new()`` carries its key and value types in the ANNOTATION, so an
    unannotated one is an object of no known class -- and a field read on it
    is answerable only by saying where the type has to be written down.
    """
    _, table = _infer(UDT_SETUP + '''
m = map.new()
missing = m.depth
''')
    found = [diag for diag in table.diags
             if diag.origin is not None and diag.origin.reason == 'unknown-class']
    assert [diag.render() for diag in found] == [
        "the class of 'm' is not known here, so its field 'depth' has no type "
        '-- annotate the value with the type it holds']


def __test_a_method_is_typed_from_the_class_it_attaches_to__():
    """``@method def bump(self: Pivot, ...)`` is a member of ``Pivot``"""
    source = UDT_SETUP + '''
@method
def bump(self: Pivot, amt: float) -> int:
    self.price += amt
    return self.idx


p = Pivot(1.0, 2, "x")
direct = bump(p, 1.0)
by_ref = method_call(bump, p, 1.0)
by_name = method_call('bump', p, 1.0)
'''
    _, table = _infer(source)
    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert types['direct'] == types['by_ref'] == types['by_name'] == INT

    sig = table.class_sigs[class_id('test', 'Pivot')]
    assert sig.fields == {'price': FLOAT, 'idx': INT, 'tag': STR}
    assert sig.methods['bump'].ret == INT
    # The receiver is the first parameter, so the signature carries the shape
    assert sig.methods['bump'].params[0] == PIVOT


def __test_a_builtin_object_has_fields_too__():
    """
    ``chart.point`` is a class like any other: its fields have declared types.

    A builtin class says what it holds in the type package rather than in a
    module interface, so the generated registry is what carries it -- without
    that, every ``p.price`` in a pitchfork or a fib script was an unknown
    that took the arithmetic built on it down with it.
    """
    types = _types(UDT_SETUP + '''
point = lib.chart.point.from_index(1, 2.0)
point_price = point.price
point_index = point.index
''')
    assert types['point'] == object_ty(builtin_class_id('ChartPoint'))
    assert types['point_price'] == FLOAT
    assert types['point_index'] == INT


def __test_a_bound_method_reference_is_an_object__():
    """``p.bump`` is a callable value; what the CALL evaluates to is the call's
    question"""
    types = _types(UDT_SETUP + '''
@method
def bump(self: Pivot, amt: float) -> int:
    return self.idx


p = Pivot(1.0, 2, "x")
ref = p.bump
''')
    assert types['ref'] == OBJECT


@pytest.mark.parametrize("expression,expected", [
    ('Pivot(1.0, 2, "x")', PIVOT),
    ('Pivot.new(1.0, 2, "x")', PIVOT),
    ('Pivot(price=1.0)', PIVOT),
    ('na(Pivot)', PIVOT),
    ('NA(Pivot)', PIVOT),
    ('box.new(1, 1.0, 2, 2.0)', object_ty(builtin_class_id('Box'))),
    ('na(lib.Box)', object_ty(builtin_class_id('Box'))),
])
def __test_a_constructed_value_knows_its_class__(expression: str, expected: str):
    """However it is spelled, what comes out is an instance of that class"""
    assert _types(UDT_SETUP + f'value = {expression}\n')['value'] == expected


def __test_a_builtin_receiver_dispatches_to_its_namespace__():
    """
    ``method_call('get_top', b)`` reaches ``box.get_top`` because ``b`` is a Box.

    This is the runtime's own dispatch rule, statically: the receiver's CLASS
    selects the implementation, and the builtin one wins over a same-named
    user function -- see ``core/pine_method.method_call``.
    """
    types = _types(UDT_SETUP + '''
b = box.new(1, 1.0, 2, 2.0)
top = method_call('get_top', b)
gone = method_call('delete', b)
a = array.new_int(3, 0)
got = method_call('get', a, 0)
''')
    assert types['top'] == FLOAT
    assert types['gone'] == VOID
    assert types['got'] == INT


def __test_a_named_user_method_still_loses_to_the_builtin__():
    """
    Handing ``method_call`` the function does not make it the one that runs.

    ``core/pine_method.method_call`` tries ``_get_builtin_method(
    method.__name__, var)`` FIRST for a callable selector too, so a user
    method named after a builtin one reaches a ``Box`` receiver only if
    ``box`` has no such name. Typing the named function outright answered
    with a type the call never produces -- here an int where the box's own
    ``delete`` returns nothing.
    """
    types = _types(UDT_SETUP + '''
@method
def delete(self: Pivot) -> int:
    return 1


b = box.new(1, 1.0, 2, 2.0)
p = Pivot(1.0, 2, "x")
builtin_wins = method_call(delete, b)
user_method = method_call(delete, p)
''')
    assert types['builtin_wins'] == VOID
    # ... while a receiver no builtin namespace answers for reaches it
    assert types['user_method'] == INT


def __test_a_user_method_on_the_builtin_class_never_displaces_it__():
    """
    Annotating the receiver with the BUILTIN class does not claim its methods.

    ``core/pine_method.method_call`` asks ``_get_builtin_method`` first in BOTH
    of its branches, so a ``@method`` declared on a ``Box`` reaches a box only
    where the ``box`` namespace has no such name. Resolving the receiver's
    class first answered with the user function instead -- an int where the
    builtin returns nothing, which is a false int-typed value.
    """
    types = _types(UDT_SETUP + '''
@method
def delete(self: Box) -> int:
    return 1


@method
def tag(self: Box) -> str:
    return "x"


b = box.new(1, 1.0, 2, 2.0)
by_name = method_call('delete', b)
by_ref = method_call(delete, b)
free_by_name = method_call('tag', b)
free_by_ref = method_call(tag, b)
''')
    assert (types['by_name'], types['by_ref']) == (VOID, VOID)
    # ... while a name the namespace does NOT have is the user's, in both forms
    assert (types['free_by_name'], types['free_by_ref']) == (STR, STR)


# --- 5. containers ---------------------------------------------------------

# Every override form the container families are typed by, at least once each.
# The element-typed reads are MEASURED on TradingView (``shape_probe1``,
# ``shape_probe3``-``shape_probe5``, FX:EURUSD@60): an ``array<int>`` answers
# int for every reduction the probes reached -- ``avg``, ``sum``, ``median``,
# ``percentile_linear_interpolation`` included -- with ``array.covariance``
# the single float exception, which is why it carries no override.
@pytest.mark.parametrize("expression,expected", [
    # 'elem0': the element of the first argument
    ('array.get(ai, 0)', INT),
    ('array.first(af)', FLOAT),
    ('array.pop(ai)', INT),
    ('array.max(ai)', INT),
    ('array.avg(ai)', INT),
    ('array.median(af)', FLOAT),
    ('array.percentile_linear_interpolation(ai, 50)', INT),
    ('matrix.get(mi, 0, 0)', INT),
    ('matrix.avg(mf)', FLOAT),
    # The one measured exception, which keeps the registry's own answer
    ('array.covariance(ai, ai)', FLOAT),
    # 'same_array0' / 'same_matrix0': the argument's whole shape comes back
    ('array.slice(ai, 0, 2)', array_of(INT)),
    ('array.copy(af)', array_of(FLOAT)),
    ('array.abs(af)', array_of(FLOAT)),
    ('matrix.transpose(mi)', matrix_of(INT)),
    ('matrix.submatrix(mf, 0, 1, 0, 1)', matrix_of(FLOAT)),
    ('map.copy(mp)', map_of(STR, FLOAT)),
    # 'array_of_arg1' / 'matrix_of_arg2': the INITIAL value types the container
    ('array.new(3, 0)', array_of(INT)),
    ('array.new(3, 0.0)', array_of(FLOAT)),
    ('array.new(3, Pivot(1.0, 2, "x"))', array_of(PIVOT)),
    ('matrix.new(2, 2, 0.0)', matrix_of(FLOAT)),
    # 'array_of_join_args': the items do, joined the way a variable would be
    ('array.from_items(1, 2, 3)', array_of(INT)),
    ('array.from_items(1, 2.0)', array_of(FLOAT)),
    # 'array_of_elem0': a row of a matrix is an array of its elements
    ('matrix.row(mi, 0)', array_of(INT)),
    ('matrix.eigenvalues(mf)', array_of(FLOAT)),
    # 'matrix_mult': matrix x matrix is a matrix, matrix x array is an array,
    # matrix x scalar is a matrix
    ('matrix.mult(mi, mi)', matrix_of(INT)),
    ('matrix.mult(mi, ai)', array_of(INT)),
    ('matrix.mult(mi, 2)', matrix_of(INT)),
    # 'map_value0' / 'array_of_map_keys0' / 'array_of_map_values0'
    ('map.get(mp, "a")', FLOAT),
    ('map.put(mp, "a", 1.0)', FLOAT),
    ('map.keys(mp)', array_of(STR)),
    ('map.values(mp)', array_of(FLOAT)),
])
def __test_every_container_form_is_element_typed__(expression: str, expected: str):
    """A container read answers the element type, which is what Pine says it is"""
    source = UDT_SETUP + f'''
ai = array.new_int(3, 0)
af = array.new_float(3, 0.0)
mi = matrix.new(2, 2, 0)
mf: Matrix[float] = matrix.new(2, 2, 0.0)
mp: dict[str, float] = map.new()
value = {expression}
'''
    assert _types('from pynecore.types import Matrix\n' + source)['value'] == expected


def __test_a_matrix_product_needs_to_know_what_it_multiplies__():
    """
    Which container comes out of ``matrix.mult`` is the SECOND operand's answer.

    A matrix by a matrix or a scalar is a matrix, a matrix by an array is an
    array -- so an operand that could be either at run time settles nothing,
    and the shape declines rather than claim the matrix one. The registry's
    own return still answers the call.
    """
    types = _types('''
from pynecore.lib import matrix

mi = matrix.new(2, 2, 0)


def main(opaque, empty: dict):
    unknown_operand = matrix.mult(mi, opaque)
    classless_operand = matrix.mult(mi, empty)
    return unknown_operand, classless_operand
''', 'main')
    assert types['unknown_operand'] == OBJECT
    assert types['classless_operand'] == OBJECT


def __test_a_container_of_unknown_element_still_answers_what_it_can__():
    """
    The shape DECLINES rather than overrides where it knows nothing.

    An override that answered from an unshaped container would replace the
    registry's own return with UNKNOWN -- so where there is no shape to read,
    the call falls through to the lib annotation exactly as before.
    """
    types = _types('''
from pynecore.lib import array


def opaque(xs):
    return array.avg(xs), array.get(xs, 0)


mean = array.avg(opaque)
''')
    # ``array.avg``'s own annotation is float; only a SHAPED argument moves it
    assert types['mean'] == FLOAT


def __test_a_nested_shape_is_read_one_layer_at_a_time__():
    """``map<string, array<float>>`` holds float arrays, and they hold floats"""
    types = _types('''
from pynecore.lib import array, map

nested: dict[str, list[float]] = {}
values = map.get(nested, "a")
first = array.get(values, 0)
keys = map.keys(nested)
one_key = array.get(keys, 0)
''')
    assert types['nested'] == map_of(STR, array_of(FLOAT))
    assert types['values'] == array_of(FLOAT)
    assert types['first'] == FLOAT
    assert types['keys'] == array_of(STR)
    assert types['one_key'] == STR


def __test_a_field_name_is_not_a_variable_of_the_scope_around_it__():
    """
    A class body declares FIELDS, and a field is reached through an instance.

    Typing the body in the enclosing scope bound every field name there too,
    so a script whose own variable happened to share a field's name met the
    field's type in it -- and two different shapes then read as a conflict
    the program does not have. Names like ``points`` or ``price`` are
    ordinary variables in a compiled script, so this was not a corner.
    """
    types = _types(UDT_SETUP + '''
points = array.new_int(3, 0)
first = array.get(points, 0)
''')
    assert types['points'] == array_of(INT)
    assert types['first'] == INT
    assert 'price' not in types


def __test_a_for_loop_takes_the_element_type__():
    """``for v in arr`` binds one element, whatever the container holds"""
    types = _types(UDT_SETUP + '''
def main():
    ai = array.new_int(3, 0)
    ps = array.new(2, Pivot(1.0, 2, "x"))
    seen = 0
    for v in ai:
        seen = v
    for p in ps:
        price = p.price
    for i, w in enumerate(ai):
        pair = i + w
    return seen
''', 'main')
    assert types['v'] == INT
    assert types['p'] == PIVOT
    assert types['price'] == FLOAT
    assert (types['i'], types['w']) == (INT, INT)


# --- 6. end to end ---------------------------------------------------------

def __test_an_element_read_pins_the_call_it_feeds__(tmp_path, monkeypatch):
    """
    The pipeline's own answer: ``array.get`` on an ``array<int>`` is int.

    ``math.max(array.get(a, 0), 1)`` is two int-typed arguments, so the site
    pins ``'ii'`` -- which it could not do while the element type was lost at
    the ``get``.
    """
    monkeypatch.syspath_prepend(tmp_path)
    path = _write(tmp_path, 'shp_pin_mod', '''"""
@pyne
"""
from pynecore.lib import array, math


def main(n: int):
    a = array.new_int(1, n)
    return math.max(array.get(a, 0), 1)
''')
    tree, _ = _analysed(path)
    call = [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'lib.math.max'][-1]

    assert [get_ty(arg) for arg in call.args] == [INT, INT]
    assert get_pin(call) == 'ii'


#: An int-TYPED element with a fractional VALUE: the one case where the static
#: answer and the runtime's value-driven one differ, so it is what proves the
#: element type reached the dispatcher.
PIN_SRC = '''
from pynecore.core.overload import overload
from pynecore.lib import array


@overload
def f(x: int) -> float:
    return 1.0


@overload
def f(x: float) -> float:
    return 2.0


def main(n: int):
    a = array.new_int(1, n)
    return f(array.get(a, 0) / 8)
'''


def _run(source: str, mod_name: str) -> dict:
    """Run the slot mini pipeline WITH the type pass and exec the result.

    :param source: Pyne-style module source
    :param mod_name: Unique module name (isolates the overload registry)
    :return: The exec'd module namespace
    """
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    layout = ModuleLayout()
    tree = PineTypeTransformer(None).visit(tree)
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {'__name__': mod_name}
    exec(compile(tree, '<shaped-type-test>', 'exec'), ns)  # noqa: S102
    return ns


def __test_the_element_type_reaches_the_dispatcher__():
    """``array.get(a, 0) / 8`` is int-typed and 1.75-valued: the int one wins"""
    ns = _run(PIN_SRC, 'shp_e2e_a')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == 1.0


def __test_the_shaped_pin_can_be_switched_off__(monkeypatch):
    """``PYNE_NO_TYPE_PIN=1`` runs the same script on value dispatch"""
    monkeypatch.setenv('PYNE_NO_TYPE_PIN', '1')
    ns = _run(PIN_SRC, 'shp_e2e_b')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    assert ns['main'](state, 14) == 2.0


def __test_two_shapes_meeting_is_reported_in_pine_spelling__():
    """A variable Pine would reject is UNKNOWN, and the message names both types"""
    _, table = _infer('''
from pynecore.lib import array


def main(flag: bool):
    a = array.new_int(1, 0)
    if flag:
        a = array.new_float(1, 0.0)
    t = array.new_int(1, 0) if flag else array.new_float(1, 0.0)
    return a, t
''')
    assert [diag.render() for diag in table.diags] == [
        "'a' gets both array<int> and array<float>, which are different types "
        '-- make both branches the same type',
        'the ternary gets both array<int> and array<float>, which are different types '
        '-- make both branches the same type',
    ]
    assert {diag.origin.reason for diag in table.diags if diag.origin is not None} \
        == {'shape-mismatch'}
    types = _types('''
from pynecore.lib import array


def main(flag: bool):
    a = array.new_int(1, 0)
    if flag:
        a = array.new_float(1, 0.0)
    return a
''', 'main')
    assert types['a'] == UNKNOWN


# --- 7. across modules -----------------------------------------------------

SHAPED_LIB = '''"""
@pyne
"""
from pynecore.core.pine_method import method
from pynecore.core.pine_udt import udt

__all__ = ['Settings', 'build', 'bump']


@udt
class Settings:
    depth: int = 10
    weights: list[float] = []


def build() -> Settings:
    return Settings(10, [])


@method
def bump(self: Settings, amt: int) -> int:
    self.depth += amt
    return self.depth
'''


def __test_a_class_travels_with_its_fields__(tmp_path, monkeypatch):
    """A dependent reads ``s.depth`` as int only because the interface says so"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'shp_lib', SHAPED_LIB)
    app = _write(tmp_path, 'shp_app', '''"""
@pyne
"""
from pynecore.core.pine_method import method_call
from pynecore.lib import array
from shp_lib import Settings, build

s: Settings = build()
depth = s.depth
weights = s.weights
first = array.get(s.weights, 0)
bumped = method_call('bump', s, 1)
''')

    _, table = _analysed(app)

    expected = object_ty(class_id(str(lib.resolve()), 'Settings'))
    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert types['s'] == expected
    assert types['depth'] == INT
    assert types['weights'] == array_of(FLOAT)
    assert types['first'] == FLOAT
    assert types['bumped'] == INT
    assert table.classes['Settings'] == class_id(str(lib.resolve()), 'Settings')


def __test_two_modules_same_named_classes_are_two_types__(tmp_path, monkeypatch):
    """Identity is (module, name): the bare name would make them one type"""
    monkeypatch.syspath_prepend(tmp_path)
    one = _write(tmp_path, 'shp_one', SHAPED_LIB)
    two = _write(tmp_path, 'shp_two', SHAPED_LIB)
    app = _write(tmp_path, 'shp_both', '''"""
@pyne
"""
from shp_one import build as build_one
from shp_two import build as build_two


def pick(flag: bool):
    s = build_one()
    if flag:
        s = build_two()
    return s
''')

    _, table = _analysed(app)

    assert table.bindings['pick']['s'].ty == UNKNOWN
    assert [diag.origin.reason for diag in table.diags if diag.origin is not None] \
        == ['shape-mismatch']
    # The two are only distinguishable because the module key is part of the id
    assert class_id(str(one.resolve()), 'Settings') \
        != class_id(str(two.resolve()), 'Settings')


#: Two libraries publishing a same-named class whose same-named field holds a
#: different scalar. The leaf says nothing about which is which.
COLLIDING_LIB = '''"""
@pyne
"""
from pynecore.core.pine_udt import udt

__all__ = ['Settings', 'build']


@udt
class Settings:
    value: int = 1


def build() -> Settings:
    return Settings(1)
'''


def __test_two_libraries_same_named_classes_stay_apart__(tmp_path, monkeypatch):
    """
    A qualified annotation is resolved by its WHOLE spelling, not by its leaf.

    ``one.Settings`` and ``two.Settings`` are two types. Keyed by the leaf, the
    second import was skipped as already known and both annotations resolved to
    whichever interface was consulted first -- so one library's field reads came
    back with the other library's type, and a false int could reach the overload
    pin of every call built on one.
    """
    monkeypatch.syspath_prepend(tmp_path)
    one = _write(tmp_path, 'shp_coll_one', COLLIDING_LIB)
    two = _write(tmp_path, 'shp_coll_two', COLLIDING_LIB.replace(
        'value: int = 1', 'value: float = 1.0').replace('Settings(1)', 'Settings(1.0)'))
    app = _write(tmp_path, 'shp_coll_app', '''"""
@pyne
"""
import shp_coll_one
import shp_coll_two
from pynecore.lib import math


def take(a: shp_coll_one.Settings, b: shp_coll_two.Settings):
    first = a.value
    second = b.value
    return math.max(a.value, 1), math.max(b.value, 1)
''')

    tree, table = _analysed(app)

    one_id = class_id(str(one.resolve()), 'Settings')
    two_id = class_id(str(two.resolve()), 'Settings')
    assert one_id != two_id
    assert table.classes['shp_coll_one.Settings'] == one_id
    assert table.classes['shp_coll_two.Settings'] == two_id

    types = {name: binding.ty for name, binding in table.bindings['take'].items()}
    assert types['a'] == object_ty(one_id)
    assert types['b'] == object_ty(two_id)
    # The field each of them declares, which is the whole point of the id
    assert (types['first'], types['second']) == (INT, FLOAT)

    # ... and the pin follows: two int-typed arguments pin as ints
    pins = [get_pin(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'lib.math.max']
    assert pins == ['ii', 'fi']


#: A library that declares a UDT, and one that declares a METHOD on it. Pine
#: lets a library extend another library's type, and the runtime finds such a
#: method by searching the modules the script imports.
BASE_LIB = '''"""
@pyne
"""
from pynecore.core.pine_udt import udt

__all__ = ['Pivot', 'build']


@udt
class Pivot:
    price: float = 0.0


def build() -> Pivot:
    return Pivot(1.0)
'''

EXTENSION_LIB = '''"""
@pyne
"""
from pynecore.core.pine_method import method
from shp_ext_base import Pivot

__all__ = ['tag']


@method
def tag(self: Pivot, extra: int) -> str:
    return "x"
'''


def __test_a_library_may_extend_another_librarys_class__(tmp_path, monkeypatch):
    """
    The receiver's class is not always where its methods live.

    ``core/pine_method.method_call`` looks in the module that defines the
    receiver's class, then in the caller's own globals, then in every library
    module the caller imported -- so a method declared on an IMPORTED class is
    part of what the declaring module publishes, and a third module calling it
    resolves through that.
    """
    monkeypatch.syspath_prepend(tmp_path)
    base = _write(tmp_path, 'shp_ext_base', BASE_LIB)
    extension = _write(tmp_path, 'shp_ext_lib', EXTENSION_LIB)
    app = _write(tmp_path, 'shp_ext_app', '''"""
@pyne
"""
from pynecore.core.pine_method import method_call
import shp_ext_lib
from shp_ext_base import Pivot, build

p: Pivot = build()
tagged = method_call('tag', p, 1)
''')

    _, table = _analysed(app)

    cid = class_id(str(base.resolve()), 'Pivot')
    # The method travels with the module that DECLARES it, keyed by the class
    # it extends -- the base library knows nothing about it
    published = registered(str(extension.resolve()))
    assert published is not None and list(published.extensions[cid]) == ['tag']
    declaring = registered(str(base.resolve()))
    assert declaring is not None and declaring.classes['Pivot'].methods == {}

    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert types['p'] == object_ty(cid)
    assert types['tagged'] == STR


def __test_a_local_method_on_an_imported_class_is_an_extension__(tmp_path, monkeypatch):
    """The calling module may declare one too, and it publishes it the same way"""
    monkeypatch.syspath_prepend(tmp_path)
    base = _write(tmp_path, 'shp_loc_base', BASE_LIB.replace('shp_ext_base', 'shp_loc_base'))
    app = _write(tmp_path, 'shp_loc_app', '''"""
@pyne
"""
from pynecore.core.pine_method import method, method_call
from shp_loc_base import Pivot, build


@method
def scale(self: Pivot, by: int) -> bool:
    return True


p: Pivot = build()
scaled = method_call('scale', p, 2)
''')

    _, table = _analysed(app)

    cid = class_id(str(base.resolve()), 'Pivot')
    assert list(table.extensions[cid]) == ['scale']
    assert table.bindings['']['scaled'].ty == BOOL


def __test_a_field_type_change_moves_the_digest__(tmp_path):
    """A field's type is part of the contract: a dependent's reads follow it"""
    def digest_of(source: str) -> str:
        # The SAME path every time: a class id carries the module key, so two
        # file names would move the digest by themselves
        path = _write(tmp_path, 'shp_digest', source)
        pine_type_artifact._registry.clear()
        sys.modules.pop('shp_digest', None)
        tree, table = _analysed(path)
        return build_interface(tree, table, str(path.resolve())).digest

    base = digest_of(SHAPED_LIB)
    changed = digest_of(SHAPED_LIB.replace('depth: int = 10', 'depth: float = 10.0'))
    body_edit = digest_of(SHAPED_LIB.replace('self.depth += amt',
                                             'self.depth += amt * 1'))

    assert base != changed
    # ... while a body edit that leaves the interface alone does not move it
    assert base == body_edit


def __test_the_artifact_round_trips_an_extension__(tmp_path, monkeypatch):
    """An extension is part of the contract, so it is in the JSON and in the digest"""
    monkeypatch.syspath_prepend(tmp_path)
    base = _write(tmp_path, 'shp_ext_base', BASE_LIB)
    path = _write(tmp_path, 'shp_json_ext', EXTENSION_LIB)
    tree, table = _analysed(path)
    interface = build_interface(tree, table, str(path.resolve()))

    cid = class_id(str(base.resolve()), 'Pivot')
    data = json.loads(json.dumps(
        table_json(tree, table, interface, path.read_bytes(), PIPELINE_DIGEST)))
    assert data['interface']['extensions'][cid]['tag']['ret'] == STR

    stat = os.stat(path)
    restored = _interface_from_json(interface.path, data, (stat.st_mtime_ns, stat.st_size))
    assert restored.extensions == interface.extensions
    assert restored.digest == interface.digest

    # A moved return type is a moved contract: every caller of the method
    # resolves through this signature
    changed = _write(tmp_path, 'shp_json_ext', EXTENSION_LIB.replace(
        '-> str:\n    return "x"', '-> int:\n    return 1'))
    pine_type_artifact._registry.clear()
    sys.modules.pop('shp_json_ext', None)
    moved_tree, moved_table = _analysed(changed)
    moved = build_interface(moved_tree, moved_table, str(changed.resolve()))
    assert moved.digest != interface.digest


def __test_the_artifact_round_trips_a_class__(tmp_path):
    """What the JSON carries is what a later process reads back"""
    path = _write(tmp_path, 'shp_json', SHAPED_LIB)
    tree, table = _analysed(path)
    interface = build_interface(tree, table, str(path.resolve()))

    data = json.loads(json.dumps(
        table_json(tree, table, interface, path.read_bytes(), PIPELINE_DIGEST)))

    published = data['interface']['classes']['Settings']
    assert published['fields'] == {'depth': INT, 'weights': array_of(FLOAT)}
    assert published['methods']['bump']['ret'] == INT

    stat = os.stat(path)
    restored = _interface_from_json(interface.path, data, (stat.st_mtime_ns, stat.st_size))
    assert restored.classes == interface.classes
    assert restored.digest == interface.digest


#: The extension library, plus an export that is not a method at all. Which
#: name an import binds is the whole question here, so there have to be two.
EXTENSION_WITH_HELPER = EXTENSION_LIB.replace(
    "from shp_ext_base import Pivot", "from shp_f5_base import Pivot").replace(
    "__all__ = ['tag']", "__all__ = ['tag', 'helper']") + '''

def helper(value: int) -> int:
    return value
'''


def __test_a_name_import_does_not_expose_a_librarys_extensions__(tmp_path, monkeypatch):
    """
    A library is within reach of a method call only where the import bound it.

    ``core/pine_method.method_call`` searches the caller's globals for the
    method, and then the MODULE objects it finds there -- so ``import ext``
    puts everything ``ext`` declares within reach, while ``from ext import
    helper`` puts ``helper`` there and nothing else. Reading every import as
    access to the whole module typed a call that ends in "No such method" at
    run time.
    """
    monkeypatch.syspath_prepend(tmp_path)
    base = _write(tmp_path, 'shp_f5_base', BASE_LIB)
    _write(tmp_path, 'shp_f5_ext', EXTENSION_WITH_HELPER)
    cid = class_id(str(base.resolve()), 'Pivot')

    def tagged(name: str, imports: str, call: str) -> str:
        """The type of a ``tag`` call under one import spelling."""
        path = _write(tmp_path, name, '"""\n@pyne\n"""\n'
                      'from pynecore.core.pine_method import method_call\n'
                      + imports + '\n'
                      'from shp_f5_base import Pivot, build\n\n'
                      'p: Pivot = build()\n'
                      'out = ' + call + '\n')
        _, table = _analysed(path)
        types = {n: binding.ty for n, binding in table.bindings[''].items()}
        assert types['p'] == object_ty(cid), 'the receiver is the library class'
        return types['out']

    # A name import binds the name, not the library behind it
    assert tagged('shp_f5_name', 'from shp_f5_ext import helper',
                  "method_call('tag', p, 1)") == UNKNOWN
    # The selector itself IS in the globals, under both spellings of the call
    assert tagged('shp_f5_sel', 'from shp_f5_ext import tag',
                  'method_call(tag, p, 1)') == STR
    assert tagged('shp_f5_str', 'from shp_f5_ext import tag',
                  "method_call('tag', p, 1)") == STR
    # A module import is what puts the whole library within reach
    assert tagged('shp_f5_mod', 'import shp_f5_ext',
                  "method_call('tag', p, 1)") == STR


#: A library whose class carries a method of its own, and a wrapper that hands
#: that class out without declaring -- or even re-exporting -- it. All the
#: caller ever receives is the class id inside the factory's return type.
BASE_WITH_METHOD = BASE_LIB.replace(
    'from pynecore.core.pine_udt import udt',
    'from pynecore.core.pine_method import method\n'
    'from pynecore.core.pine_udt import udt') + '''

@method
def area(self: Pivot, mult: int) -> int:
    return 1
'''

WRAPPER_LIB = '''"""
@pyne
"""
from shp_f6_base import Pivot, build

__all__ = ['get']


def get() -> Pivot:
    return build()
'''


def __test_a_class_is_followed_home_through_its_id__(tmp_path, monkeypatch):
    """
    A class id names the module that declares it, and that is enough to read it.

    An interface installs the classes it DECLARES, and the class a value
    carries need not be one of them: a wrapper exporting ``get() -> Pivot``
    passes another library's class through, and the caller was left holding a
    class id behind which nothing was known -- every field read of it UNKNOWN,
    every method call on it untyped. The id's module half is a source path, so
    the declaring interface is loaded on the miss, through the same machinery
    an import goes through: the dependency is recorded and the cycle guard
    still applies.
    """
    monkeypatch.syspath_prepend(tmp_path)
    base = _write(tmp_path, 'shp_f6_base', BASE_WITH_METHOD)
    wrapper = _write(tmp_path, 'shp_f6_wrap', WRAPPER_LIB)
    app = _write(tmp_path, 'shp_f6_app', '''"""
@pyne
"""
from pynecore.core.pine_method import method_call
from shp_f6_wrap import get

p = get()
price = p.price
size = method_call('area', p, 2)
''')

    _, table = _analysed(app)

    cid = class_id(str(base.resolve()), 'Pivot')
    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert types['p'] == object_ty(cid)
    assert types['price'] == FLOAT
    assert types['size'] == INT
    # The wrapper does not publish the class: the caller reaches it only
    # because the id says where it lives
    published = registered(str(wrapper.resolve()))
    assert published is not None and published.classes == {}
    # Both modules are dependencies -- the one it imports, and the one whose
    # class it ended up holding
    assert str(wrapper.resolve()) in table.deps
    assert str(base.resolve()) in table.deps


#: A library whose methods hang off the BUILTIN Box class: one name the ``box``
#: namespace also has, one it does not.
BOX_METHOD_LIB = '''"""
@pyne
"""
from pynecore.core.pine_method import method
from pynecore.types.box import Box

__all__ = ['delete', 'tag']


@method
def delete(self: Box) -> int:
    return 1


@method
def tag(self: Box) -> str:
    return "x"
'''


def __test_an_alias_does_not_rename_the_method__(tmp_path, monkeypatch):
    """
    ``from ext import delete as erase`` still runs ``box.delete`` on a box.

    The runtime looks the builtin up by ``method.__name__`` -- the name the
    DECLARING module gave the function, which no import spelling changes -- so
    a method whose name collides with a builtin one loses to it however the
    call site spells the selector. Reading the name off the spelling missed the
    collision and answered with the user function's int.
    """
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'shp_f2_ext', BOX_METHOD_LIB)
    app = _write(tmp_path, 'shp_f2_app', '''"""
@pyne
"""
from pynecore.core.pine_method import method_call
from pynecore.lib import box
from shp_f2_ext import delete as erase, tag as t

b = box.new(1, 1.0, 2, 2.0)
aliased = method_call(erase, b)
free = method_call(t, b)
''')

    _, table = _analysed(app)

    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert types['aliased'] == VOID
    # ... while an alias of a name the namespace does NOT have is the user's
    assert types['free'] == STR


def __test_a_rebound_import_is_no_longer_the_library__(tmp_path, monkeypatch):
    """
    What the globals hold AT THE CALL is what the runtime searches.

    ``method_call`` reads the caller's globals: a name the module assigns to
    holds that value and not the import, and a module alias assigned over is
    not a module at all any more, so the scan that looks for the library walks
    past it and the call ends in "No such method". Recording the import and
    never asking what became of the name typed such a call as though the
    library were still within reach.
    """
    monkeypatch.syspath_prepend(tmp_path)
    base = _write(tmp_path, 'shp_f5_base', BASE_LIB)
    _write(tmp_path, 'shp_f5_ext', EXTENSION_WITH_HELPER)
    cid = class_id(str(base.resolve()), 'Pivot')

    def tagged(name: str, imports: str) -> str:
        """The type of a ``tag`` call under one import spelling."""
        path = _write(tmp_path, name, '"""\n@pyne\n"""\n'
                      'from pynecore.core.pine_method import method_call\n'
                      + imports + '\n'
                      'from shp_f5_base import Pivot, build\n\n'
                      'p: Pivot = build()\n'
                      "out = method_call('tag', p, 1)\n")
        _, table = _analysed(path)
        types = {n: binding.ty for n, binding in table.bindings[''].items()}
        assert types['p'] == object_ty(cid), 'the receiver is the library class'
        return types['out']

    # The name the selector matched holds a number by the time the call runs
    assert tagged('shp_f5_reb_name',
                  'from shp_f5_ext import tag\ntag = 0') == UNKNOWN
    # ... and an alias assigned over is not a module to be searched
    assert tagged('shp_f5_reb_mod',
                  'import shp_f5_ext\nshp_f5_ext = 0') == UNKNOWN


def __test_an_augmented_assignment_on_a_field_reads_the_field__():
    """``p.price += 1`` is a read of the field and a write of the same type"""
    types = _types(UDT_SETUP + '''
@udt
class Pivot:
    price: float
    count: int


def main(x: int):
    p = Pivot(1.0, x)
    p.price += 1
    p.count += x
    total = p.price + p.count
    return total
''', 'main')
    assert types['total'] == FLOAT


def __test_a_class_reached_through_a_namespace_import_is_a_type__(tmp_path, monkeypatch):
    """
    ``ns.Settings`` names the class wherever a type is named: in the ``new``
    constructor Pine spells on it and in a typed ``na``, with no annotation
    having spelled it first.
    """
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'shp_ns_lib', SHAPED_LIB)
    app = _write(tmp_path, 'shp_ns_app', '''"""
@pyne
"""
from pynecore.lib import na
import shp_ns_lib as ns


def main():
    fresh = ns.Settings.new(3, [])
    empty = na(ns.Settings)
    held: ns.Settings = fresh
    return fresh.depth + held.depth
''')

    _, table = _analysed(app)

    expected = object_ty(class_id(str(lib.resolve()), 'Settings'))
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['fresh'] == expected
    assert types['empty'] == expected
    assert types['held'] == expected
    assert table.diags == []


def __test_a_value_named_like_a_class_shadows_the_constructor__():
    """``Pivot(...)`` under a parameter ``Pivot`` calls the value; ``bogus.Pivot.new`` reads a field"""
    _, table = _infer(UDT_SETUP + '''
def main(bogus: int, Pivot: float):
    fresh = bogus.Pivot.new(1.0, 1, "")
    made = Pivot(1.0, 1, "")
    return fresh
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['fresh'] == UNKNOWN
    assert types['made'] == UNKNOWN
    reasons = [diag.origin.reason for diag in table.diags if diag.origin is not None]
    assert reasons.count('unknown-call') == 2, [diag.message for diag in table.diags]


def _reasons(table: PineTypeTable) -> list[str]:
    return [diag.origin.reason for diag in table.diags if diag.origin is not None]


def _lines(table: PineTypeTable) -> list[int]:
    """Diagnostic lines counted from the end of ``UDT_SETUP``."""
    base = UDT_SETUP.count('\n')
    return [diag.line - base for diag in table.diags]


def __test_a_field_store_is_checked_like_a_read__():
    """``p.x = v`` needs the field to exist and the value to fit what it holds"""
    _, table = _infer(UDT_SETUP + '''
def main(x: int, f: float):
    p = Pivot(1.0, x, "a")
    p.nosuch = 3
    p.idx = f
    p.price = x
    p.tag = na
    x.y = 1
    return p
''')
    assert _reasons(table) == ['unknown-field', 'type-mismatch', 'unknown-field']
    assert _lines(table) == [4, 5, 8]
    assert "'Pivot' has no field 'nosuch'" in table.diags[0].message
    assert "holds int and is assigned float" in table.diags[1].message
    assert "a int has no field 'y'" in table.diags[2].message


def __test_a_bare_object_cannot_take_a_field_store__():
    """An object of no known class is a complaint about the receiver's typing"""
    _, table = _infer(UDT_SETUP + '''
def main():
    o = enumerate(array.new_int(0))
    o.x = 1
    return o
''')
    assert _reasons(table) == ['unknown-class']


def __test_a_history_read_needs_a_series__():
    """``x[1]`` reads a series' history: a plain scalar name has no bars behind it"""
    source = '''
from pynecore import lib
from pynecore.types import Series


def main(p: Series[float], q: float):
    x = 5
    s: Series[int] = 5
    d: Series
    d = q * 2
    a = x[1]
    b = s[1]
    c = lib.close[1]
    e = p[1]
    g = d[1]
    h = q[1]
    i = lib.ta.sma(lib.close, 3)[1]
    return a + b + c + e + g + h + i
'''
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    table = infer_module(tree, 'probe.py')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == UNKNOWN and types['h'] == UNKNOWN and types['i'] == UNKNOWN
    assert types['b'] == INT and types['c'] == FLOAT and types['e'] == FLOAT
    assert types['g'] == FLOAT
    assert _reasons(table) == ['not-series', 'not-series', 'not-series']
    assert [diag.line for diag in table.diags] == [11, 16, 17]
    assert 'declare it Series[int]' in table.diags[0].fix


def __test_a_container_is_not_indexed__():
    """An element read is ``array.get``; a Python index on the container is not Pine"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = array.new_float(0)
    b = array.new_int(0)
    return a[0] + b[0]
''')
    assert _reasons(table) == ['not-pine', 'not-pine']
    assert 'array.get' in table.diags[0].fix


def __test_a_scalar_is_not_iterable__():
    """A for loop walks a range or an array, never a number"""
    _, table = _infer(UDT_SETUP + '''
def main(n: int):
    total = 0
    for i in lib.close:
        total += 1
    for j in range(n):
        total += j
    for k in array.new_int(0):
        total += k
    return total
''')
    assert _reasons(table) == ['not-iterable']
    assert _lines(table) == [4]


def __test_a_user_call_is_held_to_its_signature__():
    """Too many, too few, unknown or repeated arguments are the call's error"""
    _, table = _infer(UDT_SETUP + '''
def f(a, b):
    return a + b


def g(a: int):
    return a


def main():
    r1 = f(1)
    r2 = f(1, 2, 3)
    r3 = f(1, nosuch=2)
    r4 = f(1, a=2)
    r5 = g("s")
    r6 = g(1.5)
    r7 = g(1)
    r8 = f(1, 2)
    return r1
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['r7'] == INT and types['r8'] == INT
    assert all(types[name] == UNKNOWN for name in ('r1', 'r2', 'r3', 'r4', 'r5', 'r6'))
    assert _reasons(table) == ['bad-call'] * 6
    messages = [diag.message for diag in table.diags]
    assert "'f' needs an argument for 'b'" in messages[0]
    assert "takes 2 positional argument(s), 3 passed" in messages[1]
    assert "'f' has no parameter 'nosuch'" in messages[2]
    assert "'a' is passed to 'f' twice" in messages[3]
    assert "'g' takes int for 'a', string passed" in messages[4]
    assert "'g' takes int for 'a', float passed" in messages[5]


def __test_a_lib_call_is_held_to_its_signature__():
    """The registry says what each lib function takes; the call has to meet it"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = lib.ta.sma(lib.close)
    b = lib.ta.sma("s", 5)
    c = lib.ta.sma(lib.close, 5.0)
    d = lib.ta.sma(lib.close, length=5)
    e = lib.ta.sma(lib.close, nosuch=5)
    f = lib.math.max(1, 2, 3)
    g = lib.plot(lib.close, title="x", color=lib.color.red)
    lib.fill(g, g, color=lib.color.red, title="band")
    return a
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['d'] == FLOAT and types['f'] == INT
    assert all(types[name] == UNKNOWN for name in ('a', 'b', 'c', 'e'))
    assert _reasons(table) == ['bad-call'] * 4
    messages = [diag.message for diag in table.diags]
    assert "'ta.sma' does not take 1 argument(s)" in messages[0]
    assert "'ta.sma' takes float for 'source', string passed" in messages[1]
    assert "'ta.sma' takes int for 'length', float passed" in messages[2]
    assert "'ta.sma' has no parameter 'nosuch'" in messages[3]


def __test_an_element_put_has_to_fit_the_array__():
    """``array.push(a, v)`` puts v in: it has to be what the array holds"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = array.new_float(0)
    array.push(a, 1)
    array.set(a, 0, 2.5)
    array.push(a, na)
    array.push(a, "s")
    return array.get(a, 0)
''')
    assert _reasons(table) == ['shape-mismatch']
    assert _lines(table) == [7]
    assert table.pins_suppressed is not None
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == UNKNOWN


def __test_an_element_put_of_the_wrong_kind_is_the_only_finding__():
    """``array.insert`` and ``array.unshift`` are puts too; a fitting one is silent"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = array.new_int(0)
    array.unshift(a, 1)
    array.insert(a, 0, 2)
    array.fill(a, 3)
    return array.get(a, 0)
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == array_of(INT)


def __test_the_profile_builtins_have_types__():
    """``max``/``min`` are math's, ``print`` is a statement, ``enumerate`` an iterable"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = max(1, 2)
    b = min(1, 2.0)
    print("hi")
    for i, x in enumerate(array.new_int(0)):
        a += i
    return a + b
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == INT and types['b'] == FLOAT


def __test_a_bare_wrapper_declaration_of_an_unknown_is_one_finding__():
    """``x: Series = nosuch`` is the value's problem, reported once"""
    _, table = _infer('''
from pynecore.types import Series


def main():
    x: Series = nosuch
    return x
''')
    assert _reasons(table) == ['unknown-name']
    binding = table.binding('main', 'x')
    assert binding is not None and binding.unknown is not None
    assert (binding.unknown.line, binding.unknown.col) == (6, 16)


def __test_every_series_head_gives_a_history__():
    """``PersistentSeries`` and ``IBPersistentSeries`` are series slots too"""
    source = '''
from pynecore import lib
from pynecore.types import PersistentSeries, IBPersistentSeries, Persistent


def main():
    s: PersistentSeries[float] = lib.na(float)
    u: IBPersistentSeries[int] = 0
    p: Persistent[float] = 0.0
    a = s[1]
    b = u[1]
    c = p[1]
    return a + b + c
'''
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    table = infer_module(tree, 'probe.py')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == FLOAT and types['b'] == INT and types['c'] == UNKNOWN
    assert _reasons(table) == ['not-series']


def __test_a_lib_constant_that_is_a_string_fits_a_string_parameter__():
    """``format.percent`` is a Format, and a Format is what Pine calls a const string"""
    _, table = _infer(UDT_SETUP + '''
def main():
    p = lib.plot(lib.close, format=lib.format.percent, precision=2)
    m = matrix.new(2, 2, 0.0)
    matrix.sort(m, 0, lib.order.descending)
    t = lib.ticker.new(lib.syminfo.prefix, lib.syminfo.ticker, lib.session.regular)
    return p
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['t'] == STR


def __test_a_bar_offset_is_an_int__():
    """``ta.highestbars`` counts bars, so ``bar_index + ta.highestbars(...)`` stays int"""
    _, table = _infer(UDT_SETUP + '''
def use(idx: int):
    return idx


def main():
    hb = lib.bar_index + lib.ta.highestbars(lib.high, 20)
    lb = lib.ta.lowestbars(20)
    a = array.new_float(0)
    v = array.get(a, lib.ta.highestbars(lib.high, 20))
    return use(hb) + use(lb) + v
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['hb'] == INT and types['lb'] == INT and types['v'] == FLOAT


def __test_a_user_overload_group_is_held_to_its_implementations__():
    """A call no implementation takes is an error before any pin"""
    _, table = _infer(UDT_SETUP + '''
@overload
def f(a: int) -> int:
    return a


@overload
def f(a: int, b: int) -> int:
    return a + b


def main():
    r1 = f()
    r2 = f(1, 2, 3, 4)
    r3 = f(1, nosuch=2)
    r4 = f("s")
    r5 = f(1, 2)
    r6 = f(1)
    return r1
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['r5'] == INT and types['r6'] == INT
    assert all(types[name] == UNKNOWN for name in ('r1', 'r2', 'r3', 'r4'))
    assert _reasons(table) == ['bad-call'] * 4
    assert all("no overload of 'f'" in diag.message for diag in table.diags)


def __test_a_constructor_is_held_to_its_fields__():
    """A UDT's fields are its constructor's parameters, defaults included"""
    _, table = _infer(UDT_SETUP + '''
@udt
class Pair:
    price: float
    count: int = 0


def main():
    a = Pair(1.0, 2, 3)
    b = Pair.new(nosuch=1)
    c = Pair(price="s")
    d = Pair()
    e = Pair(1.0, price=2.0)
    f = Pair(1.0)
    g = Pair.new(1, count=2)
    h = Pair(na, 3)
    return a
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert all(types[name] == UNKNOWN for name in ('a', 'b', 'c', 'd', 'e'))
    assert all(types[name] == object_ty(class_id('test', 'Pair')) for name in ('f', 'g', 'h'))
    assert _reasons(table) == ['bad-call'] * 5
    messages = [diag.message for diag in table.diags]
    assert "'Pair' has 2 field(s), 3 argument(s) passed" in messages[0]
    assert "'Pair' has no field 'nosuch'" in messages[1]
    assert "'Pair.price' holds float, string passed" in messages[2]
    assert "'Pair' needs a value for 'price'" in messages[3]
    assert "'price' is passed to 'Pair' twice" in messages[4]


def __test_a_map_and_a_matrix_put_are_checked_like_an_array_put__():
    """What goes into a container has to be what the container holds"""
    _, table = _infer(UDT_SETUP + '''
def main():
    mp: dict[str, float] = map.new()
    map.put(mp, "a", 1)
    prev = map.put(mp, "b", 2.5)
    m = matrix.new(2, 2, 0.0)
    matrix.set(m, 0, 0, 1)
    matrix.fill(m, 2)
    matrix.add_row(m, 0, array.new_int(2, 0))
    return prev
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['prev'] == FLOAT and types['m'] == matrix_of(FLOAT)
    assert types['mp'] == map_of(STR, FLOAT)
    for put in ('map.put(mp, "a", "x")', 'map.put(mp, 1, 1.0)', 'matrix.set(m, 0, 0, "x")',
                'matrix.fill(m, "x")', 'matrix.add_row(m, 0, array.new_string(0))'):
        _, table = _infer(UDT_SETUP + f'''
def main():
    mp: dict[str, float] = map.new()
    m = matrix.new(2, 2, 0.0)
    {put}
    return map.get(mp, "a") + matrix.get(m, 0, 0)
''')
        assert _reasons(table) == ['shape-mismatch'], put
        assert table.pins_suppressed is not None, put


def __test_a_lib_value_is_not_called__():
    """``close(1)`` reaches nothing; a module property may be called and reads the same"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = lib.close(1)
    b = lib.timeframe.period()
    c = lib.time()
    return b
''')
    assert _reasons(table) == ['unknown-lib']
    assert "'close' is a lib value" in table.diags[0].message
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == UNKNOWN and types['b'] == STR and types['c'] == INT


def __test_a_put_without_an_operand_puts_nothing__():
    """``matrix.add_row(m)`` appends an empty row; the array operand is optional"""
    _, table = _infer(UDT_SETUP + '''
def main():
    m = matrix.new(0, 3, 0.0)
    matrix.add_row(m)
    matrix.add_row(m, 0)
    matrix.add_col(m)
    return matrix.get(m, 0, 0)
''')
    assert not table.diags and table.pins_suppressed is None
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['m'] == matrix_of(FLOAT)


def __test_a_builtin_reached_by_the_method_spelling_is_held_to_its_shape__():
    """``method_call('get', a, 0)`` IS ``array.get(a, 0)``, shape included"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = array.new_float(0)
    x = method_call('get', a)
    y = method_call('get', a, 0, 1, 2)
    z = method_call('get', a, 0)
    method_call('push', a, "s")
    return z
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['x'] == UNKNOWN and types['y'] == UNKNOWN and types['z'] == FLOAT
    assert _reasons(table) == ['bad-call', 'bad-call', 'shape-mismatch']


def __test_a_rewritten_lib_name_keeps_its_position__():
    """A diagnostic about ``close`` points at where ``close`` was written"""
    _, table = _infer(UDT_SETUP + '''
from pynecore.lib import close


def main():
    a = array.new_int(0)
    array.concat(a, close)
    return a
''')
    assert _reasons(table) == ['shape-mismatch']
    base = UDT_SETUP.count('\n')
    assert (table.diags[0].line - base, table.diags[0].col) == (7, 20)


def __test_the_compiled_loop_and_history_forms_are_typed__():
    """``inline_series(expr, n)`` is ``expr[n]``; ``pine_loop`` is the counter object,
    whose ``value`` is typed from the bounds like a ``for`` variable and whose
    ``step`` is the loop test"""
    tree, table = _infer(UDT_SETUP + '''
from pynecore.core.series import inline_series
from pynecore import pine_loop
from pynecore.lib import close


def main(length: int):
    h = inline_series(close, 1)
    k = inline_series(length, 2)
    __loop_1__ = pine_loop(1, 2)
    while __loop_1__.step(length):
        i = __loop_1__.value
    __loop_2__ = pine_loop(0, 1)
    while __loop_2__.step(2.5):
        j = __loop_2__.value
    return h + k + i + j
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['h'] == FLOAT and types['k'] == INT
    assert types['__loop_1__'] == PINE_LOOP and types['__loop_2__'] == PINE_LOOP
    assert types['i'] == INT and types['j'] == FLOAT
    steps = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'step']
    assert len(steps) == 2 and all(get_ty(step) == BOOL for step in steps)


def __test_a_loop_bound_that_is_not_a_number_is_reported__():
    """A string bound is a bad call, at the call; the counter it leaves untyped says so"""
    _, table = _infer('''
from pynecore import pine_loop


def main():
    __loop_1__ = pine_loop("a", 1)
    while __loop_1__.step(3):
        i = __loop_1__.value
    return i
''')
    assert _reasons(table) == ['bad-call', 'unknown-value']
    assert "'pine_loop' takes a number for a bound, string passed" in table.diags[0].message
    assert [diag.line for diag in table.diags] == [6, 8]


def __test_a_compiled_library_export_proxy_is_a_callable_reference__(tmp_path, monkeypatch):
    """``name: _Protocol = Exported()`` is what a compiled library binds each export to
    at module level; it is nothing to report, and an importer reads the export's own
    signature through it"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'shp_r6_addlib', '''
"""
@pyne edge
"""
from pynecore.core.pine_cast import cast_int
from pynecore.core.pine_export import Exported, export
from pynecore.lib import plot, script
from typing import Protocol, Any


__all__ = ['addOne', 'twice']


class _ProtocolAddone(Protocol):
    def __call__(self, x: int) -> Any: ...


class _ProtocolTwice(Protocol):
    def __call__(self, v: float) -> Any: ...


addOne: _ProtocolAddone = Exported()
twice: _ProtocolTwice = Exported()


@script.library("addlib")
def main():
    @export
    def addOne(x: int):
        return x + 1

    @export
    def twice(v: float):
        return addOne(cast_int(v)) * 2

    plot(addOne(1))
''')
    path = _write(tmp_path, 'shp_r6_user', '''
"""
@pyne
"""
import shp_r6_addlib as addlib
from shp_r6_addlib import twice
from pynecore.lib import plot


def main():
    a = addlib.addOne(1)
    b = twice(2.5)
    plot(a + b)
''')
    _, library = _analysed(tmp_path / 'shp_r6_addlib.py')
    assert not library.diags
    _, table = _analysed(path)
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == INT and types['b'] == INT


def __test_a_put_takes_an_unchecked_operand__():
    """``None``, a bare object and a typeless na go into a container unchecked, the way a
    declaration takes them; an UNKNOWN operand stays a report"""
    _, table = _infer(UDT_SETUP + '''
from pynecore.types import Map


def main():
    a = array.new_int(0)
    array.push(a, None)
    array.push(a, lib.math.floor)
    m: Map[int, int] = map.new()
    map.put(m, None, 1)
    x = matrix.new(1, 1, 0)
    matrix.add_row(x, None)
    return a
''')
    assert not table.diags
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == array_of(INT)
    _, table = _infer(UDT_SETUP + '''
def main(n):
    a = array.new_int(0)
    array.push(a, n)
    return a
''')
    assert 'unknown-value' in _reasons(table)
    assert table.bindings['main']['a'].ty == UNKNOWN


def __test_a_failing_put_in_value_position_is_reported_once__():
    """The operand that does not fit is the one cause, whether the put's result is used
    and however the put is spelled"""
    _, table = _infer(UDT_SETUP + '''
def main():
    a = array.new_int(0)
    x = array.push(a, "s")
    b = array.new_int(0)
    y = method_call('push', b, "s")
    return x, y
''')
    assert _reasons(table) == ['shape-mismatch', 'shape-mismatch']
    assert _lines(table) == [4, 6]


def __test_a_function_selector_naming_a_group_is_typed_like_the_direct_call__():
    """``method_call(scale, 2, 3)`` selects the implementation the receiver and the
    arguments pin, without stamping the plumbing; a call no implementation takes is a
    bad call of the METHOD, and one the group cannot settle statically reads as the
    direct spelling does, named after the method"""
    tree, table = _infer(UDT_SETUP + '''
from pynecore.core.overload import overload


@overload
def getType(this: int) -> str:
    return "int"


@overload
def getType(this: float) -> str:
    return "float"


@overload
def scale(this: int, by: int) -> int:
    return this * by


@overload
def scale(this: float, by: float) -> float:
    return this * by


def main():
    d = 1.0
    x = method_call(getType, d)
    y = method_call(scale, 2, 3)
    z = method_call(scale, 2.0, 3.0)
    w = method_call(scale, "s", 3)
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['x'] == STR and types['y'] == INT
    assert types['z'] == UNKNOWN and types['w'] == UNKNOWN
    assert sorted(_reasons(table)) == ['bad-call', 'unknown-return']
    messages = sorted(diag.message for diag in table.diags)
    assert messages == ["no overload of 'scale' takes these arguments",
                        "the call to method 'scale' has no known type"]
    plumbing = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'method_call']
    assert len(plumbing) == 4 and all(get_pin(node) is None for node in plumbing)


def __test_a_method_call_that_reaches_no_method_names_the_method__():
    """The plumbing is always there: what is missing is a method of the receiver"""
    _, table = _infer(UDT_SETUP + '''
def bump(this: int) -> int:
    return this + 1


bump = 3


def main(n):
    p = Pivot(1.0, 2)
    a = method_call('copy', p)
    b = method_call(bump, 1)
    c = method_call('copy', n)
    return a, b, c
''')
    # ``n`` is reported as an unannotated parameter; the method call on it says
    # nothing new, its receiver's UNKNOWN carries the provenance
    assert _reasons(table) == ['unknown-method', 'rebound-name', 'unannotated-param']
    assert table.diags[0].message == "'copy' is not a method of Pivot here"
    assert table.diags[1].message == "'bump' is assigned as well as defined, so what it calls is unknown"
    assert _lines(table)[:2] == [11, 12]
    assert not any('method_call' in diag.message for diag in table.diags)


def __test_a_split_persistent_series_declaration_keeps_its_position__(tmp_path):
    """``a: PersistentSeries[T] = v`` is split into a Persistent and a Series half; a
    diagnostic on either half points at the line the user wrote, never at 0"""
    path = _write(tmp_path, 'shp_r6_pseries', '''
"""
@pyne
"""
from pynecore.lib import na, plot
from pynecore.types import Label, Line, PersistentSeries, Series


def main():
    a: PersistentSeries[Label] = na(Line)
    b: Series[Label] = na(Line)
    plot(1)
''')
    _, table = _analysed(path)
    mismatches = [diag for diag in table.diags
                  if diag.origin is not None and diag.origin.reason == 'type-mismatch']
    assert [(diag.line, diag.col) for diag in mismatches] == [(10, 4), (11, 4)]
    assert all(diag.line for diag in table.diags)
