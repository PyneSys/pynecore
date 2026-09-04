"""
Pine's tuple is a type, not a bag: ``[float, int]`` is as much a shape as
``array<int>`` is.

A tuple is the only Pine type that holds SEVERAL types at once, and every
script that returns two values through one function -- 59 returns and 47
unpacks in the corpus alone -- passes through it. Reading it as an anonymous
object stopped the types at the unpack: ``[middle, upper, lower] = ta.bb(...)``
gave three unknowns, and everything computed from them was unknown too.

So the tuple joins the grammar as ``T:`` followed by one length-prefixed item
per position. The length prefix is what makes it work at all: the other shapes
are self-delimiting only because each has exactly ONE tail, and a class id ends
in a filesystem path that may hold any character -- commas, colons, ``#``,
spaces. An item that carries its own length needs no escaping and no lookahead,
so a shape of any depth round-trips unchanged.

Everything else follows the shapes that came before: ``head()`` is ``'o'``, so
the arithmetic and the pin wire format never learn about tuples; the join is
elementwise, which is what makes ``[na, na]`` take the types of the branch it
meets; and two tuples of different arity are a Pine ERROR, reported where they
meet rather than silently widened.
"""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

from pynecore.core.import_hook import PIPELINE_DIGEST, analyse_source
from pynecore.transformers import pine_type_artifact
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.pine_type_artifact import (
    build_interface, registered, table_json, _interface_from_json,
)
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    BOOL, COLOR, FLOAT, INT, OBJECT, STR, TYPELESS, UNKNOWN, VOID,
    annotation_type, arity, array_of, builtin_class_id, class_id, elements_of, get_pin,
    get_ty, head, is_shaped, is_tuple, join, map_of, matrix_of, object_ty, render_ty,
    shape_mismatch, tuple_of,
)
from pynecore.transformers.pine_type_table import PineTypeTable


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
        if name.startswith('tup_'):
            del sys.modules[name]


def _infer(source: str) -> tuple[ast.Module, PineTypeTable]:
    """Infer a snippet the way the pipeline does, import normalization first."""
    tree = ImportNormalizerTransformer().visit(ast.parse(source))
    return tree, infer_module(tree, 'test')


def _types(source: str, scope: str = '') -> dict[str, str]:
    """The bindings of one scope, as name -> type."""
    _, table = _infer(source)
    return {name: binding.ty for name, binding in table.bindings.get(scope, {}).items()}


def _diags(source: str) -> list[str]:
    """Every diagnostic message one snippet produces."""
    _, table = _infer(source)
    return [diag.message for diag in table.diags]


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


#: A path with every character that could break a naive encoding: a comma, a
#: colon, a space, brackets and the ``#`` a class id is itself spelled with.
AWKWARD: str = class_id('/tmp/a b, c:d(#e)/mod.py', 'Pivot')

#: What a script writes at the top of a tuple test.
SETUP = '''
from pynecore import lib
from pynecore.lib import array, close, math, na, ta
from pynecore.types import Series


'''


# --- 1. the grammar --------------------------------------------------------


@pytest.mark.parametrize("elements", [
    [INT, FLOAT],
    [FLOAT, FLOAT, FLOAT],
    [TYPELESS, TYPELESS],
    [UNKNOWN, INT],
    [object_ty(AWKWARD), INT],
    [array_of(object_ty(AWKWARD)), map_of(STR, tuple_of([INT, object_ty(AWKWARD)]))],
    [tuple_of([INT, tuple_of([FLOAT, BOOL])]), matrix_of(object_ty(AWKWARD)), COLOR],
])
def __test_a_tuple_round_trips_whatever_it_holds__(elements: list[str]):
    """
    Any element shape survives being put next to another one.

    This is the whole reason for the length prefix: a class id ends in a source
    PATH, which may hold commas, colons and even the ``#`` the id is spelled
    with, so no separator could tell one element from the next.
    """
    ty = tuple_of(elements)
    assert is_tuple(ty)
    assert list(elements_of(ty)) == elements
    assert arity(ty) == len(elements)
    # ... and it is an object to everything that reads the lattice
    assert head(ty) == OBJECT
    assert is_shaped(ty)


def __test_a_tuple_of_nothing_is_not_a_tuple__():
    """An empty sequence is no Pine value, and nothing reads it as one"""
    assert tuple_of([]) == OBJECT
    assert elements_of(OBJECT) == () and arity(OBJECT) == 0
    assert elements_of(array_of(INT)) == ()
    assert not is_tuple(array_of(tuple_of([INT, INT])))
    # A malformed encoding is not a tuple this module wrote
    assert elements_of('T:2:i') == () and elements_of('T:x:i') == ()


@pytest.mark.parametrize("ty,rendered", [
    (tuple_of([INT, FLOAT]), '[int, float]'),
    (tuple_of([FLOAT, INT]), '[float, int]'),
    (tuple_of([TYPELESS, TYPELESS]), '[na, na]'),
    (tuple_of([array_of(INT), object_ty(AWKWARD)]), '[array<int>, Pivot]'),
    (array_of(tuple_of([INT, STR])), 'array<[int, string]>'),
])
def __test_a_tuple_renders_the_way_pine_spells_one__(ty: str, rendered: str):
    """The message a user reads says ``[int, float]``, not the encoding"""
    assert render_ty(ty) == rendered


@pytest.mark.parametrize("left,right,expected", [
    # Position by position, which is the same rule one level down
    (tuple_of([INT, INT]), tuple_of([FLOAT, INT]), tuple_of([FLOAT, INT])),
    (tuple_of([INT, STR]), tuple_of([INT, STR]), tuple_of([INT, STR])),
    # A typeless element takes the other side's type, one position at a time
    (tuple_of([TYPELESS, TYPELESS]), tuple_of([FLOAT, INT]), tuple_of([FLOAT, INT])),
    (tuple_of([TYPELESS, INT]), tuple_of([STR, TYPELESS]), tuple_of([STR, INT])),
    # One position disagreeing says nothing about the other
    (tuple_of([INT, STR]), tuple_of([INT, FLOAT]), tuple_of([INT, UNKNOWN])),
    # Two arities are two types
    (tuple_of([INT, INT]), tuple_of([INT, INT, INT]), UNKNOWN),
    # ... and so are a tuple and anything that is not one
    (tuple_of([INT, INT]), FLOAT, UNKNOWN),
    (tuple_of([INT, INT]), array_of(INT), UNKNOWN),
    # The typeless ``na`` itself is still typeless against a tuple
    (tuple_of([INT, INT]), TYPELESS, tuple_of([INT, INT])),
])
def __test_two_tuples_join_position_by_position__(left: str, right: str, expected: str):
    """A branch says something about every position, not about the value as a whole"""
    assert join(left, right) == expected
    assert join(right, left) == expected


@pytest.mark.parametrize("left,right,conflict", [
    (tuple_of([INT, INT]), tuple_of([INT, INT, INT]), True),
    (tuple_of([INT, INT]), FLOAT, True),
    (tuple_of([INT, INT]), OBJECT, True),
    (tuple_of([INT, INT]), UNKNOWN, False),
    (tuple_of([INT, INT]), TYPELESS, False),
    (tuple_of([INT, INT]), tuple_of([INT, INT]), False),
])
def __test_a_tuple_conflict_is_a_conflict_with_everything__(left: str, right: str,
                                                            conflict: bool):
    """
    A tuple is the one shape that also disagrees with the scalars.

    ``[float, int]`` and ``float`` are two types: a function returning one on
    one path and the other on another is rejected by Pine, and saying so is
    more use than an unexplained unknown.
    """
    assert shape_mismatch(left, right) is conflict
    assert shape_mismatch(right, left) is conflict


@pytest.mark.parametrize("spelling,expected", [
    ('tuple[int, float]', tuple_of([INT, FLOAT])),
    ('tuple[PyneFloat, PyneInt]', tuple_of([FLOAT, INT])),
    ('tuple[Series[int], list[float]]', tuple_of([INT, array_of(FLOAT)])),
    ('tuple[float, float, float]', tuple_of([FLOAT, FLOAT, FLOAT])),
    # A sequence of unknown length is not a Pine tuple: its arity is what an
    # unpack is checked against
    ('tuple[int, ...]', OBJECT),
    ('tuple', OBJECT),
    # One element is still a fixed arity: ``tuple[int]`` unpacks with one name
    ('tuple[int]', tuple_of([INT])),
])
def __test_an_annotation_spells_a_tuple_too__(spelling: str, expected: str):
    """``tuple[float, int]`` is how a lib function declares Pine's ``[float, int]``"""
    assert annotation_type(ast.parse(spelling, mode='eval').body) == expected


# --- 2. the engine ---------------------------------------------------------


def __test_a_tuple_literal_is_a_tuple__():
    """Both sequence literals PyneComp emits a Pine tuple as are one"""
    types = _types(SETUP + '''
i = 2
f = 1.5
pair = (i, f)
listed = [i, f]
nested = (i, (f, "x"))
empty = []
single = (i,)
''')
    assert types['pair'] == tuple_of([INT, FLOAT])
    assert types['listed'] == tuple_of([INT, FLOAT])
    assert types['nested'] == tuple_of([INT, tuple_of([FLOAT, STR])])
    # The empty list is what a lower-timeframe security read defaults to: an
    # array, and never a tuple
    assert types['empty'] == OBJECT
    assert types['single'] == tuple_of([INT])


def __test_an_untyped_na_stays_typeless_inside_a_tuple__():
    """
    ``__block_result__ = (na, na)`` is a real emitted form, and it types nothing.

    The elements are TYPELESS, so the branch that does produce values decides
    every position -- which is how a block whose first assignment is the na
    filler still hands back a typed pair.
    """
    types = _types(SETUP + '''
def main(flag: bool, src: float):
    __block_result__ = (na, na)
    if flag:
        __block_result__ = (src, 2)
    first, second = __block_result__
    return first
''', 'main')
    assert types['__block_result__'] == tuple_of([FLOAT, INT])
    assert (types['first'], types['second']) == (FLOAT, INT)


def __test_an_unpack_gives_every_name_its_own_type__():
    """The point of the whole shape: each position lands on its own name"""
    types = _types(SETUP + '''
def pair(x: int):
    return x, x * 1.0


a, b = pair(2)
c, d = (1, "x")
''')
    assert (types['a'], types['b']) == (INT, FLOAT)
    assert (types['c'], types['d']) == (INT, STR)


def __test_an_unpack_of_the_wrong_length_is_reported__():
    """
    Pine matches the arity of an unpack against the type, and so does this.

    Guessing at an alignment would put one position's type on another
    position's name, so both names are unknown and the mismatch is named where
    it stands.
    """
    source = SETUP + '''
def pair(x: int):
    return x, x + 1


a, b, c = pair(2)
'''
    types = _types(source)
    assert (types['a'], types['b'], types['c']) == (UNKNOWN, UNKNOWN, UNKNOWN)
    assert any('2 elements' in message and 'unpacks 3' in message
               for message in _diags(source))


def __test_an_unpack_of_a_value_that_is_no_tuple_says_nothing__():
    """An unknown value is not a wrong one: nothing is claimed, nothing reported"""
    source = SETUP + '''
def opaque(x):
    return x


a, b = opaque(2)
'''
    types = _types(source)
    assert (types['a'], types['b']) == (UNKNOWN, UNKNOWN)
    assert _diags(source) == []


def __test_a_starred_target_is_not_pine__():
    """Pine has no form for it, so the names are unknown and it is reported"""
    source = SETUP + '''
def pair(x: int):
    return x, x + 1


a, *rest = pair(2)
'''
    types = _types(source)
    assert (types['a'], types['rest']) == (UNKNOWN, UNKNOWN)
    assert any('starred' in message for message in _diags(source))


def __test_a_constant_index_picks_a_position__():
    """
    A tuple has no history: ``t[0]`` is a position, not a bar back.

    Only a constant index is known before the program runs, which is the only
    form Pine could have anyway.
    """
    types = _types(SETUP + '''
def pair(x: int):
    return x, x * 1.0


t = pair(2)
first = t[0]
second = t[1]
past_end = t[2]
computed = t[lib.bar_index]
''')
    assert (types['first'], types['second']) == (INT, FLOAT)
    assert (types['past_end'], types['computed']) == (UNKNOWN, UNKNOWN)


def __test_a_function_returns_the_tuple_it_builds__():
    """The return type is the shape, and two returns join position by position"""
    _, table = _infer(SETUP + '''
def pair(flag: bool, x: int):
    if flag:
        return x, 1.5
    return x + 1, 2
''')
    assert table.funcs['pair'].ret == tuple_of([INT, FLOAT])


def __test_two_returns_of_different_shape_are_reported__():
    """A tuple against a scalar, and two arities, are both Pine errors"""
    source = SETUP + '''
def mixed(flag: bool, x: int):
    if flag:
        return x, x
    return x


def wide(flag: bool, x: int):
    if flag:
        return x, x
    return x, x, x
'''
    _, table = _infer(source)
    assert table.funcs['mixed'].ret == UNKNOWN
    assert table.funcs['wide'].ret == UNKNOWN
    messages = [diag.message for diag in table.diags]
    assert any('[int, int]' in message and 'int' in message for message in messages)
    assert any('[int, int, int]' in message for message in messages)
    assert all(diag.origin is not None and diag.origin.reason == 'shape-mismatch'
               for diag in table.diags)


def __test_a_ternary_over_tuples_joins_elementwise__():
    """The arms of a ternary meet the way two returns do"""
    types = _types(SETUP + '''
def main(flag: bool, src: float):
    picked = (src, 1) if flag else (1.0, 2)
    both, count = picked
    return both
''', 'main')
    assert types['picked'] == tuple_of([FLOAT, INT])
    assert (types['both'], types['count']) == (FLOAT, INT)


def __test_iterating_a_tuple_is_not_pine__():
    """Pine iterates arrays, not tuples: the loop variable is simply unknown"""
    types = _types(SETUP + '''
def main(x: int):
    pair = (x, x + 1)
    for item in pair:
        held = item
    return x
''', 'main')
    assert types['item'] == UNKNOWN
    assert types['held'] == UNKNOWN


def __test_the_pin_never_sees_a_tuple__():
    """
    The wire format is untouched: a tuple is an object to the pin.

    What the unpack does reach is the site AFTER it -- an int-typed half of a
    pair pins the call it feeds, which is the whole point of typing the halves
    apart.
    """
    tree, _ = _infer(SETUP + '''
def pair(x: int):
    return x, x + 1


t = pair(2)
a, b = pair(2)
whole = math.max(t, 1)
half = math.max(a, 1)
''')
    pins = [get_pin(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'lib.math.max']
    # The tuple itself pins nothing -- it is an object, and an object argument
    # can never be the one an overload turns on
    assert pins == [None, 'ii']


# --- 3. the lib table ------------------------------------------------------


def __test_a_lib_tuple_return_unpacks_into_its_parts__():
    """
    ``ta.bb`` and friends declare ``tuple[...]``, and the registry carries it.

    Five lib functions return a Pine tuple, and every corpus script using one
    lost all of it at the unpack before -- ``[middle, upper, lower] =
    ta.bb(...)`` gave three unknowns.
    """
    types = _types(SETUP + '''
middle, upper, lower = ta.bb(close, 20, 2.0)
trend, direction = ta.supertrend(3.0, 10)
''')
    assert (types['middle'], types['upper'], types['lower']) == (FLOAT, FLOAT, FLOAT)
    # ... and the int half of a mixed pair is an INT-typed value, which is what
    # the whole pass exists for
    assert (types['trend'], types['direction']) == (FLOAT, INT)


def __test_a_lib_tuple_half_pins_what_it_feeds__():
    """An unpacked int reaches an overload site as an int"""
    tree, _ = _infer(SETUP + '''
trend, direction = ta.supertrend(3.0, 10)
picked = math.max(direction, 1)
''')
    pins = [get_pin(node) for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'lib.math.max']
    assert pins == ['ii']


# --- 4. request.security ---------------------------------------------------


def __test_a_security_tuple_arrives_as_a_tuple__(tmp_path):
    """
    The two halves of a security read are typed apart.

    ``SecurityTransformer`` splits the call into a guarded ``__sec_write__``
    holding the tuple EXPRESSION and a ``__sec_read__`` whose default is an
    N-tuple of ``na``. The write says what each position is, the typeless
    default says nothing, and the join over the two lands each position on its
    own name.
    """
    path = _write(tmp_path, 'tup_sec', '''"""
@pyne
"""
from pynecore.lib import high, request, script, syminfo


@script.indicator("t")
def main():
    hi, count = request.security(syminfo.tickerid, "D", (high, 1))
    return hi
''')

    _, table = _analysed(path)

    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert (types['hi'], types['count']) == (FLOAT, INT)
    assert table.diags == []


# --- 5. across modules -----------------------------------------------------


def __test_an_exported_tuple_return_travels__(tmp_path, monkeypatch):
    """A declared tuple return is part of the contract, and the caller unpacks it"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'tup_lib', '''"""
@pyne
"""

__all__ = ['split', 'guess']


def split(x: int) -> tuple[int, float]:
    return x, x * 1.0


def guess(x):
    return x, x * 1.0
''')
    app = _write(tmp_path, 'tup_app', '''"""
@pyne
"""
from tup_lib import guess, split

a, b = split(2)
c, d = guess(2)
''')

    _, table = _analysed(app)

    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert (types['a'], types['b']) == (INT, FLOAT)
    # The unannotated export is still typed from its DECLARATION and nothing
    # else: a caller's arguments cannot reach into a module compiled on its own
    assert (types['c'], types['d']) == (UNKNOWN, UNKNOWN)
    assert any(diag.origin is not None and diag.origin.reason == 'unannotated-import'
               for diag in table.diags)


def __test_a_tuple_element_type_moves_the_digest__(tmp_path):
    """An element's type is part of the contract: a dependent's unpack follows it"""
    def digest_of(source: str) -> str:
        path = _write(tmp_path, 'tup_digest', source)
        pine_type_artifact._registry.clear()
        sys.modules.pop('tup_digest', None)
        tree, table = _analysed(path)
        return build_interface(tree, table, str(path.resolve())).digest

    before = digest_of('"""\n@pyne\n"""\n\n\ndef split(x: int) -> tuple[int, float]:\n'
                       '    return x, x * 1.0\n')
    after = digest_of('"""\n@pyne\n"""\n\n\ndef split(x: int) -> tuple[float, float]:\n'
                      '    return x * 1.0, x * 1.0\n')
    assert before != after


def __test_the_artifact_round_trips_a_tuple__(tmp_path):
    """What the JSON carries is what a later process reads back"""
    path = _write(tmp_path, 'tup_json', '''"""
@pyne
"""

__all__ = ['split']


def split(x: int) -> tuple[int, float]:
    return x, x * 1.0
''')
    tree, table = _analysed(path)
    interface = build_interface(tree, table, str(path.resolve()))

    data = json.loads(json.dumps(
        table_json(tree, table, interface, path.read_bytes(), PIPELINE_DIGEST)))
    assert data['interface']['exports']['split']['ret'] == tuple_of([INT, FLOAT])

    stat = os.stat(path)
    restored = _interface_from_json(interface.path, data, (stat.st_mtime_ns, stat.st_size))
    assert restored.exports == interface.exports
    assert restored.digest == interface.digest


def __test_a_class_field_may_be_a_tuple__(tmp_path, monkeypatch):
    """A shape is a shape wherever it is declared, fields included"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = _write(tmp_path, 'tup_cls', '''"""
@pyne
"""
from pynecore.core.pine_udt import udt

__all__ = ['Holder', 'build']


@udt
class Holder:
    pair: tuple[int, float] = (0, 0.0)


def build() -> Holder:
    return Holder((1, 1.0))
''')
    app = _write(tmp_path, 'tup_cls_app', '''"""
@pyne
"""
from tup_cls import Holder, build

h: Holder = build()
held = h.pair
first, second = h.pair
''')

    _, table = _analysed(app)

    cid = class_id(str(lib.resolve()), 'Holder')
    published = registered(str(lib.resolve()))
    assert published is not None
    assert published.classes['Holder'].fields['pair'] == tuple_of([INT, FLOAT])
    types = {name: binding.ty for name, binding in table.bindings[''].items()}
    assert types['h'] == object_ty(cid)
    assert types['held'] == tuple_of([INT, FLOAT])
    assert (types['first'], types['second']) == (INT, FLOAT)


# --- 8. what a tuple is NOT -------------------------------------------------


def __test_a_tuple_is_no_container_for_the_container_calls__():
    """``array.copy`` of a tuple hands back a list, which the tuple's positions say nothing about"""
    types = _types(SETUP + '''
from pynecore.lib import matrix, map as pmap


def main(x: int):
    t = (x, "x")
    u = array.copy(t)
    v = array.concat(t, t)
    w = matrix.transpose(t)
    ai = array.new_int(2, x)
    k = pmap.copy(ai)
    first = u[0]
    picked = math.max(first, 1)
    return picked
''', 'main')
    assert types['t'] == tuple_of([INT, STR])
    for name in ('u', 'v', 'w', 'k'):
        assert not is_tuple(types[name]), name
        assert types[name] != array_of(INT), name
    # Nothing downstream is typed from the tuple's positions any more: the
    # list may have been reversed by the time ``u[0]`` is read, so neither
    # the int nor the string of the literal may show up, and no pin either
    assert types['first'] not in (INT, STR)
    assert types['picked'] == UNKNOWN


def __test_a_tuple_of_one_keeps_its_shape__():
    """``tuple[int]`` is a one-element tuple, unpacked with one name"""
    assert annotation_type(ast.parse('tuple[int]', mode='eval').body, {}) == tuple_of([INT])
    assert annotation_type(ast.parse('tuple[int, ...]', mode='eval').body, {}) == OBJECT
    _, table = _infer(SETUP + '''
def one(x: int) -> tuple[int]:
    return (x,)


def main(x: int):
    a, = one(x)
    b = (x * 2,)
    return a
''')
    assert table.funcs['one'].ret == tuple_of([INT])
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['a'] == INT
    assert types['b'] == tuple_of([INT])
    assert not [diag for diag in table.diags if diag.origin is not None
                and diag.origin.reason == 'shape-mismatch']


def __test_a_tuple_of_one_travels_between_modules__(tmp_path, monkeypatch):
    """A one-element tuple return is the same contract across an import"""
    monkeypatch.syspath_prepend(tmp_path)
    _write(tmp_path, 'tup_one_lib', '''"""
@pyne
"""

__all__ = ['one']


def one(x: int) -> tuple[int]:
    return (x,)
''')
    app = _write(tmp_path, 'tup_one_app', '''"""
@pyne
"""
from tup_one_lib import one

a, = one(2)
''')
    _, table = _analysed(app)
    assert table.bindings['']['a'].ty == INT


def __test_a_conflict_inside_a_tuple_is_still_reported__():
    """Two branches disagreeing in ONE position are a Pine error, wherever the position is"""
    _, table = _infer(SETUP + '''
def pair(flag: bool, x: int):
    if flag:
        return array.new_int(1, x), x
    return array.new_float(1, x), x
''')
    ret = table.funcs['pair'].ret
    # The clean position keeps its type: the unpack that follows still needs it
    assert is_tuple(ret) and elements_of(ret)[1] == INT
    assert elements_of(ret)[0] == UNKNOWN
    shape_diags = [diag for diag in table.diags
                   if diag.origin is not None and diag.origin.reason == 'shape-mismatch']
    assert shape_diags, [diag.message for diag in table.diags]
    assert 'array<int>' in shape_diags[0].message and 'array<float>' in shape_diags[0].message


def __test_a_container_operation_checks_its_second_operand__():
    """``array.concat`` mutates its receiver, so an operand that does not fit spoils it"""
    _, table = _infer(SETUP + '''
from pynecore.lib import matrix


def main(x: int, y: float):
    ai = array.new_int(1, x)
    af = array.new_float(1, y)
    t = ("x",)
    array.concat(ai, t)
    last = array.last(ai)
    picked = math.max(last, 1)

    good = array.new_float(1, y)
    array.concat(good, array.new_int(1, x))
    kept = array.last(good)

    narrow = array.new_int(1, x)
    array.concat(narrow, af)
    lost = array.last(narrow)

    mf = matrix.new(1, 1, y)
    matrix.sum(mf, 1)
    mi = matrix.new(1, 1, x)
    matrix.sum(mi, 1.5)
    return picked
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['ai'] == UNKNOWN
    assert types['last'] == UNKNOWN
    assert types['picked'] == UNKNOWN
    assert types['good'] == array_of(FLOAT) and types['kept'] == FLOAT
    assert types['narrow'] == UNKNOWN and types['lost'] == UNKNOWN
    assert types['mf'] == matrix_of(FLOAT)
    assert types['mi'] == UNKNOWN
    reasons = [diag.origin.reason for diag in table.diags if diag.origin is not None]
    assert reasons.count('shape-mismatch') == 3, [diag.message for diag in table.diags]
    messages = [diag.message for diag in table.diags]
    assert any('array<int>' in m and '[string]' in m for m in messages)
    assert any('array<int>' in m and 'array<float>' in m for m in messages)
    assert any('matrix<int>' in m and 'float' in m for m in messages)


def __test_an_unknown_operand_spoils_the_receiver_too__():
    """What was appended is not known, so what the array holds is not known either"""
    _, table = _infer(SETUP + '''
def main(x: int):
    ai = array.new_int(1, x)
    array.concat(ai, getattr(x, "elsewhere"))
    return array.last(ai)
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    assert types['ai'] == UNKNOWN
    # The operand's own failure is the one report: a node is diagnosed once,
    # and the root cause outranks what it spoils
    assert [(diag.origin.reason, diag.origin.detail) for diag in table.diags
            if diag.origin is not None] == [('unknown-call', 'getattr')]
    # ... and the failed merge still takes every pin away
    assert table.pins_suppressed is not None


def __test_a_spoiled_container_takes_every_pin_with_it__():
    """An alias of the receiver is the same list, so no type downstream may drive a dispatch"""
    from pynecore.transformers.pine_type_rules import get_pins, get_vector

    tree, table = _infer(SETUP + '''
def main(x: int):
    ai = array.new_int(1, x)
    fine = math.max(x, 1)
    alias = ai
    array.concat(alias, ("x",))
    bad = array.last(ai)
    picked = math.max(bad, 1)
    return picked
''')
    types = {name: binding.ty for name, binding in table.bindings['main'].items()}
    # The alias itself is invalidated; the other name for the same list is not
    # tracked, and its stale type is exactly what may not reach a pin
    assert types['alias'] == UNKNOWN
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert all(get_pin(node) is None and get_pins(node) is None and get_vector(node) is None
               for node in calls)
    assert all(site.pin is None for site in table.calls)
    assert table.pins_suppressed is not None
    assert table.pins_suppressed.origin is not None \
        and table.pins_suppressed.origin.reason == 'shape-mismatch'

    # The same script without the spoiled merge pins ``math.max(x, 1)`` as usual
    tree, table = _infer(SETUP + '''
def main(x: int):
    ai = array.new_int(1, x)
    fine = math.max(x, 1)
    return fine
''')
    assert table.pins_suppressed is None
    assert any(get_pin(node) == 'ii' for node in ast.walk(tree) if isinstance(node, ast.Call))


def __test_a_spoiled_module_takes_its_importers_pins_too__(tmp_path, monkeypatch):
    """A return type published over a spoiled container may not drive a dispatch anywhere"""
    from pynecore.transformers.pine_type_rules import get_pins, get_vector

    monkeypatch.syspath_prepend(tmp_path)
    lib_path = _write(tmp_path, 'tup_spoiled', '''"""
@pyne
"""
from pynecore.lib import array

__all__ = ['poisoned']


def poisoned(x: int):
    ai = array.new_int(1, x)
    alias = ai
    array.concat(alias, ("x",))
    return array.last(ai)
''')
    app = _write(tmp_path, 'tup_spoiled_app', '''"""
@pyne
"""
from pynecore.lib import math
from tup_spoiled import poisoned

__all__ = ['wrapped']


def wrapped(x: int):
    return poisoned(x)


picked = math.max(poisoned(1), 1)
fine = math.max(1, 2)
''')
    top = _write(tmp_path, 'tup_spoiled_top', '''"""
@pyne
"""
from pynecore.lib import math
from tup_spoiled_app import wrapped

other = math.max(wrapped(1), 1)
''')

    tree, table = _analysed(app)
    published = registered(str(lib_path))
    assert published is not None and published.suppressed
    # The importer is reported at the import, and gives every pin up
    assert table.pins_suppressed is not None
    assert table.pins_suppressed.origin is not None \
        and table.pins_suppressed.origin.reason == 'suppressed-import'
    assert 'tup_spoiled' in table.pins_suppressed.message
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert calls and all(get_pin(node) is None and get_pins(node) is None
                         and get_vector(node) is None for node in calls)

    # ... and so does whoever imports the importer: the contract carries it
    tree, table = _analysed(top)
    assert table.pins_suppressed is not None
    assert all(get_pin(node) is None for node in ast.walk(tree) if isinstance(node, ast.Call))

    # The artifact carries the flag and the digest moves with it
    clean = build_interface(ast.parse('"""\n@pyne\n"""\n'), PineTypeTable(), str(lib_path))
    assert published.digest != clean.digest
    data = json.loads(json.dumps(
        table_json(tree, table, published, lib_path.read_bytes(), PIPELINE_DIGEST)))
    assert data['interface']['suppressed'] == published.suppressed
    rebuilt = _interface_from_json(str(lib_path), data, (0, -1))
    assert rebuilt.suppressed == published.suppressed
    assert rebuilt.digest == published.digest
