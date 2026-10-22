"""The native ``pine_type_rules`` port (``rust/crates/pyne-transform``) against the Python module.

Every ported function is called through ``_native._rules`` and compared with its Python
original: exhaustively over the type alphabet and the annotation and operator forms,
and over every call the Python pipeline makes while it transforms the parity corpus of
``test_135_native_transform``.
"""
import ast
import itertools
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore.core import import_hook
from pynecore.transformers import factory_fields
from pynecore.transformers import pine_type_rules as rules

from .test_135_native_transform import __test_helper_inputs, __test_helper_require_native

#: The scalar characters and a few class ids, a module path full of punctuation included
__test_helper_SCALARS = ['i', 'f', 'b', 's', 'c', 'o', 'v', '?', '*']
__test_helper_CLASSES = [
    'o:lib#Line', 'o:lib#PineLoop', 'o:lib#Box', 'o:lib#HLine', 'o:/x/zigzag.py#Pivot',
    'o:/Users/Ádám/a, b:c#d.py#Pivot', 'o:x', 'o:', 'o:lib#', 'o:lib#Table',
]

#: Strings the grammar does not produce, read the way the Python module reads them
__test_helper_MALFORMED = [
    '', 'a:', 'm:', 'T:', 'T:x', 'T:1:', 'T:5:ab', 'T:01:i', 'T:1:i1:f', 'T:1:iX', 'T:0:',
    'T:00:', 'T:0:1:i', 'oo', 'ab', 'é', 'T:99999999999999999999999:i', 'T:1:é', 'T:2:é',
    'M:sx', 'M:é:i', 'M:s:', 'a:é', 'T:-1:i', 'T: 1:i', 'T:1:i:', 'O', 'I', 'F',
]

#: Every ``ast.operator`` and ``ast.unaryop`` class name
__test_helper_BINARY = sorted(cls.__name__ for cls in ast.operator.__subclasses__())
__test_helper_UNARY = sorted(cls.__name__ for cls in ast.unaryop.__subclasses__())


def __test_helper_native(name, *args):
    from pynecore.transformers import _native

    return _native._rules(name, *args)


def __test_helper_unsupported() -> type[Exception]:
    from pynecore.transformers import _native

    return _native.Unsupported


def __test_helper_types() -> list[str]:
    """The type alphabet: scalars, classes, one and two levels of shapes, tuples, garbage."""
    base = __test_helper_SCALARS + __test_helper_CLASSES
    shaped = [f'{prefix}{ty}' for prefix in ('a:', 'm:') for ty in base]
    shaped += [f'M:{key}:{value}' for key in __test_helper_SCALARS for value in base]
    nested = [f'{prefix}{ty}' for prefix in ('a:', 'm:') for ty in shaped[:40]]
    nested += [f'M:s:{ty}' for ty in shaped[:40]]
    pool = __test_helper_SCALARS + __test_helper_CLASSES[:5] + ['a:i', 'a:f', 'm:i', 'M:s:a:f']
    tuples = [rules.tuple_of(list(items)) for items in itertools.product(pool, repeat=2)]
    tuples += [rules.tuple_of([ty]) for ty in pool]
    tuples += [rules.tuple_of([outer, inner]) for outer in tuples[:20] for inner in pool]
    tuples += [rules.tuple_of(['i', 'f', 'b']), rules.tuple_of(['i', 'f', 'o:/a#B'])]
    return base + shaped + nested + tuples + __test_helper_MALFORMED


def __test_helper_pair_types() -> list[str]:
    """A smaller alphabet for the functions of two types."""
    types = __test_helper_types()
    picked = __test_helper_SCALARS + __test_helper_CLASSES + __test_helper_MALFORMED[:12]
    picked += [ty for ty in types if ty.startswith(('a:', 'm:'))][:30]
    picked += [ty for ty in types if ty.startswith('M:')][:20]
    picked += [ty for ty in types if ty.startswith('T:')][:80]
    return list(dict.fromkeys(picked))


def __test_helper_outcome(function, *args):
    """What a Python function answers, or the exception class it raises."""
    try:
        return 'ok', function(*args)
    except Exception as err:  # noqa: BLE001 -- any raise is an outcome to skip
        return 'raised', type(err).__name__


def __test_helper_plain(value):
    """A Python result as the bridge spells it: tuples and lists of strings as tuples."""
    if isinstance(value, (list, tuple)):
        return tuple(__test_helper_plain(item) for item in value)
    return value


def __test_helper_check(name, function, args_list, stats: Counter):
    """Compare one function over argument tuples; a Python raise is skipped and counted."""
    for args in args_list:
        kind, expected = __test_helper_outcome(function, *args)
        if kind == 'raised':
            stats[f'{name} (Python raises)'] += 1
            continue
        native_args = [list(arg) if isinstance(arg, tuple) else arg for arg in args]
        got = __test_helper_native(name, *native_args)
        assert __test_helper_plain(got) == __test_helper_plain(expected), (name, args)
        stats[name] += 1


def __test_native_rule_tables_match__():
    """The constants, sets and tables are the Python module's, entry by entry and in order."""
    __test_helper_require_native()
    for name, value in __test_helper_native('constants').items():
        assert getattr(rules, name) == value, name
    for name, value in __test_helper_native('sets').items():
        assert set(getattr(rules, name)) == set(value) and len(value) == len(set(value)), name
    tables = __test_helper_native('tables')
    for name in ('LIB_TYPE_OVERRIDES', 'BUILTIN_CALL_TYPES', 'BUILTIN_NAME_TYPES',
                 'ANNOTATION_TYPES', 'BUILTIN_NAMESPACES', 'OVERRIDE_PARAM_NAMES',
                 'BUILTIN_CLASSES', 'PIN_CONTAINERS'):
        assert list(tables[name].items()) == list(getattr(rules, name).items()), name
    assert {key: sorted(value) for key, value in tables['NA_TEXT_PARAMS'].items()} \
        == {key: sorted(value) for key, value in rules.NA_TEXT_PARAMS.items()}
    assert list(tables['NA_TEXT_PARAMS']) == list(rules.NA_TEXT_PARAMS)
    assert set(map(tuple, tables['NOT_PINE_STRINGS'])) == set(rules.NOT_PINE_STRINGS)

    names = list(rules.LIB_TYPE_OVERRIDES) + list(rules.BUILTIN_CALL_TYPES) + [
        'nope', 'math', '', 'input', 'run', '__name__', 'Line', 'Pivot', 'plot', 'title']
    for name in names:
        assert __test_helper_native('lib_type_override', name) == rules.LIB_TYPE_OVERRIDES.get(name)
        assert __test_helper_native('builtin_call_type', name) == rules.BUILTIN_CALL_TYPES.get(name)
        assert __test_helper_native('builtin_name_type', name) == rules.BUILTIN_NAME_TYPES.get(name)
        assert (__test_helper_native('override_param_names', name)
                == rules.OVERRIDE_PARAM_NAMES.get(name))
        assert __test_helper_native('builtin_class', name) == rules.BUILTIN_CLASSES.get(name)
        for param in ('title', 'char', 'text', 'message', 'order', 'sort_field', 'x'):
            assert (__test_helper_native('is_na_text_param', name, param)
                    == (param in rules.NA_TEXT_PARAMS.get(name, ())))
    for callee, param in itertools.product(
            ['matrix.sort', 'array.sort', 'array.sort_indices', 'plot'], ['order', 'sort_field']):
        assert (__test_helper_native('is_not_pine_string', callee, param)
                == ((callee, param) in rules.NOT_PINE_STRINGS))


def __test_native_lattice_matches__():
    """Every type function over the type alphabet, alone and in pairs."""
    __test_helper_require_native()
    stats = Counter()
    types = __test_helper_types()
    singles = [(ty,) for ty in types]
    for name in ('head', 'is_shaped', 'is_array', 'is_matrix', 'is_map', 'is_tuple', 'arity',
                 'elements_of', 'class_of', 'element_of', 'key_of', 'value_of', 'render_ty',
                 'namespace_of', 'array_of', 'matrix_of', 'object_ty', 'builtin_class_id',
                 'is_int_typed'):
        __test_helper_check(name, getattr(rules, name), singles, stats)
    for name, members in (('is_numeric', rules.NUMERIC), ('is_known', rules.KNOWN),
                          ('is_scalar', rules.SCALARS)):
        __test_helper_check(name, lambda ty, m=members: ty in m, singles, stats)
    __test_helper_check('is_na_widen', rules.is_na_widen,
                        [(None,)] + singles + [(kind,) for kind in 'fsamntAMXx'], stats)
    __test_helper_check('tuple_of', rules.tuple_of,
                        [((),)] + [(tuple(items),) for items in
                                   itertools.product(types[:25], repeat=2)], stats)
    pairs = list(itertools.product(__test_helper_pair_types(), repeat=2))
    for name in ('join', 'shape_mismatch', 'shape_conflict', 'compare_type', 'map_of',
                 'class_id'):
        __test_helper_check(name, getattr(rules, name), pairs, stats)
    for ty in types:
        assert __test_helper_native('elements_of', rules.tuple_of(rules.elements_of(ty))) \
            == tuple(rules.elements_of(rules.tuple_of(rules.elements_of(ty))))
    assert stats['join'] > 10_000, stats


def __test_native_operators_match__():
    """``binop_type`` over every operator and ``unaryop_type`` over every unary one."""
    __test_helper_require_native()
    stats = Counter()
    operands = __test_helper_SCALARS + __test_helper_CLASSES[:3] + [
        'a:i', 'm:f', 'M:s:i', rules.tuple_of(['i', 'f']), '', 'ii', 'é']
    for op_name in __test_helper_BINARY:
        op = getattr(ast, op_name)()
        for left, right in itertools.product(operands, repeat=2):
            assert (__test_helper_native('binop_type', op_name, left, right)
                    == rules.binop_type(op, left, right)), (op_name, left, right)
            stats['binop_type'] += 1
    for op_name in __test_helper_UNARY:
        op = getattr(ast, op_name)()
        for operand in __test_helper_types():
            assert (__test_helper_native('unaryop_type', op_name, operand)
                    == rules.unaryop_type(op, operand)), (op_name, operand)
            stats['unaryop_type'] += 1
    for value in (None, True, False, 0, 1, 10 ** 40, -3, 1.5, float('nan'), '', 'x', b'x', 1j,
                  ..., (1, 2), frozenset({1})):
        assert __test_helper_native('constant_type', value) == rules.constant_type(value), value
    assert len(__test_helper_BINARY) == 13 and len(__test_helper_UNARY) == 4


#: Annotation atoms: names, dotted names and constants, forward references included
__test_helper_ANNOTATION_NAMES = [
    'int', 'PyneInt', 'float', 'PyneFloat', 'bool', 'PyneBool', 'str', 'PyneStr', 'Color',
    'None', 'Series', 'PersistentSeries', 'NA', 'Persistent', 'IBPersistent',
    'IBPersistentSeries', 'Optional', 'list', 'Array', 'SequenceView', 'Matrix', 'dict', 'Map',
    'tuple', 'Tuple', 'set', 'Line', 'Box', 'Label', 'PineLoop', 'Pivot', 'Settings', 'Any',
    'object', 'NoneType', 'Union', 'foo', 'string', 'color',
]
__test_helper_ANNOTATION_ATOMS = __test_helper_ANNOTATION_NAMES + [
    'lib.Line', 'm.Settings', 'a.Settings', 'x.NA', 'typing.Optional', 'typing.Union',
    'lib.Matrix', 'lib.PyneInt', 'chr.SessionTimeRange', 'a.b.c', 'typing.Any', 'm.list',
    'None', '1', '1.5', 'True', '...', "'int'", "'list[Pivot]'", "'int | None'", "' int'",
    "'1 +'", "'NA'", "'yield'", "'tuple[int, ...]'", "b'x'", "\"'int'\"", "'Optional[int]'",
    "'m.Settings'", "''", "'int;'", "'1if x else 2'", "'list[int]  '", "'a: int'", "f'x'",
    'x()', '[int]', '(int)', '-1',
]
__test_helper_ANNOTATION_HEADS = __test_helper_ANNOTATION_NAMES + [
    'typing.Optional', 'typing.Union', 'lib.Matrix', 'pine.Series', 'a.b.list', 'x()']

#: Two class maps: none, and one that shadows a builtin class and a container name
__test_helper_CLASS_MAPS = [
    {},
    {'Pivot': '/x/zigzag.py#Pivot', 'Settings': '/m.py#Settings', 'm.Settings': '/m.py#Settings',
     'Line': '/mine.py#Line', 'list': '/x.py#list', 'chr.SessionTimeRange': 'lib#SessionTimeRange',
     'a.b.c': '/Ádám/a.py#c'},
]


def __test_helper_annotations() -> list[str]:
    """Annotation sources: atoms, unions, subscripts of every head, nested shapes."""
    atoms = __test_helper_ANNOTATION_ATOMS
    out = list(atoms)
    out += [f'{a} | {b}' for a, b in itertools.product(atoms, repeat=2)]
    out += [f'{a} | {b} | {c}' for a, b, c in itertools.product(atoms[:12], atoms[::5], atoms[3:9])]
    out += [f'{head}[{a}]' for head in __test_helper_ANNOTATION_HEADS for a in atoms]
    pairs = list(itertools.product(atoms[::3], atoms[1::4]))
    for head in __test_helper_ANNOTATION_HEADS:
        out += [f'{head}[{a}, {b}]' for a, b in pairs]
        out += [f'{head}[{a}, ...]' for a in atoms[:6]]
        out += [f'{head}[int, float, str]', f'{head}[()]', f'{head}[(int, float)]']
    out += [
        'dict[str, list[float]]', 'Map[PyneStr, Matrix[int]]', 'list[dict[int, list[Pivot]]]',
        'tuple[list[int], Matrix[float], dict[str, int]]', 'Series[list[int]] | NA',
        'tuple[tuple[int, float], str]', "list['Pivot']", "dict[str, 'list[float]']",
        'Optional[int] | None', 'Union[int, None]', 'Union[int]', 'Union[NoneType, int]',
        'NA[int] | None', 'NA | None', 'None | NA', 'int | float | str | bool | NA',
        'Series[float] | float | NA[float]', 'list[int | float]', 'dict[str | int, float]',
        "'int' | None", "'Any' | int", 'Series', 'Persistent[float | int]',
    ]
    return list(dict.fromkeys(out))


def __test_helper_compare_annotation(node, classes, stats: Counter, where: str):
    """One annotation: its type under the class map and whether it takes ``None``."""
    unsupported = __test_helper_unsupported()
    expected = rules.annotation_type(node, classes)
    try:
        got = __test_helper_native('annotation_type', node, dict(classes))
    except unsupported:
        # Only a forward reference outside what the native parser provably agrees on
        assert any(isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                   for sub in ast.walk(node)), (where, ast.dump(node))
        stats['annotation_type unsupported'] += 1
    else:
        assert got == expected, (where, ast.dump(node), classes)
        stats['annotation_type'] += 1
    expected = rules.annotation_takes_none(node)
    try:
        got = __test_helper_native('annotation_takes_none', node)
    except unsupported:
        stats['annotation_takes_none unsupported'] += 1
    else:
        assert got == expected, (where, ast.dump(node))
        stats['annotation_takes_none'] += 1
    assert __test_helper_native('bare_wrapper', node) == rules.bare_wrapper(node)


@pytest.mark.filterwarnings('ignore::SyntaxWarning')
def __test_native_annotations_match__():
    """``annotation_type`` and its family over every annotation form, both class maps."""
    __test_helper_require_native()
    stats = Counter()
    assert __test_helper_native('annotation_type', None, {}) == rules.annotation_type(None)
    assert __test_helper_native('annotation_takes_none', None) == rules.annotation_takes_none(None)
    for source in __test_helper_annotations():
        try:
            node = ast.parse(source, mode='eval').body
        except SyntaxError:
            continue
        for classes in __test_helper_CLASS_MAPS:
            __test_helper_compare_annotation(node, classes, stats, source)
    print(dict(stats))
    assert stats['annotation_type'] > 40_000, stats
    # The forward references the native parser reads are the common ones
    for source in ("'int'", "'list[Pivot]'", "'int | None'", "'tuple[int, ...]'", "'NA'"):
        node = ast.parse(source, mode='eval').body
        assert (__test_helper_native('annotation_type', node, __test_helper_CLASS_MAPS[1])
                == rules.annotation_type(node, __test_helper_CLASS_MAPS[1]))


#: The parameter types the overload selection is checked over
__test_helper_PARAM_TYPES = [
    'i', 'f', 'b', 's', 'c', 'o', '?', '*', 'a:i', 'a:f', 'a:?', 'm:i', 'm:f', 'm:?', 'o:x#A',
    'o:x#B', 'M:s:i', 'T:1:i1:f', 'a:b', 'm:s']
__test_helper_PIN_CHARS = 'ifbsc*IFBS1234o'


def __test_helper_impl(rng: random.Random) -> rules.ImplSig:
    count = rng.randint(0, 3)
    params = tuple(rng.choice(__test_helper_PARAM_TYPES) for _ in range(count))
    fits = ''.join(rng.choice('yn?') for _ in range(count + rng.choice((0, 0, 0, 1, 2))))
    return rules.ImplSig(params=params, required=rng.randint(0, count),
                         open_ended=rng.random() < 0.1, ret=rng.choice(__test_helper_PARAM_TYPES),
                         fits=fits, names=('x', 'y', 'z')[:count])


def __test_helper_impl_tuples(impls):
    return [tuple(impl) for impl in impls]


def __test_helper_index(impls, choice):
    if choice is None:
        return None
    return next(index for index, impl in enumerate(impls) if impl is choice)


def __test_native_overload_selection_matches__():
    """The pin, the fits and the static overload selection over random groups."""
    __test_helper_require_native()
    rng = random.Random(136)
    for args in itertools.product(__test_helper_PARAM_TYPES + ['', 'é'], repeat=2):
        assert __test_helper_native('pin_chars', list(args)) == rules.pin_chars(list(args))
        assert __test_helper_native('pin_for', list(args)) == rules.pin_for(list(args))
        assert (__test_helper_native('overload_result', list(args))
                == rules.overload_result(list(args)))
    assert __test_helper_native('pin_for', []) == rules.pin_for([])
    assert __test_helper_native('overload_result', []) == rules.overload_result([])
    defaults = [None, '0'] + __test_helper_PARAM_TYPES
    for declared, default, takes in itertools.product(__test_helper_PARAM_TYPES, defaults,
                                                      (False, True)):
        assert (__test_helper_native('default_fit', declared, default, takes)
                == rules.default_fit(declared, default, takes)), (declared, default, takes)

    answered = 0
    for _ in range(40_000):
        impls = [__test_helper_impl(rng) for _ in range(rng.randint(1, 4))]
        width = rng.randint(0, 4)
        pin = ''.join(rng.choice(__test_helper_PIN_CHARS) for _ in range(width))
        shapes = [rng.choice(__test_helper_PARAM_TYPES) for _ in range(rng.randint(0, width))]
        native_impls = __test_helper_impl_tuples(impls)
        choice = rules.overload_choice(impls, pin, shapes)
        assert (__test_helper_native('overload_choice', native_impls, pin, shapes)
                == __test_helper_index(impls, choice)), (impls, pin, shapes)
        assert (__test_helper_native('overload_pick', native_impls, pin, shapes)
                == rules.overload_pick(impls, pin, shapes))
        names = rng.sample(['x', 'y', 'z', 'k'], rng.randint(0, 2))
        kw_chars = {name: rng.choice(__test_helper_PIN_CHARS) for name in names}
        kw_shapes = {name: rng.choice(__test_helper_PARAM_TYPES) for name in names
                     if rng.random() < 0.7}
        choice = rules.overload_binding(impls, pin, shapes, kw_chars, kw_shapes)
        assert (__test_helper_native('overload_binding', native_impls, pin, shapes, kw_chars,
                                     kw_shapes)
                == __test_helper_index(impls, choice)), (impls, pin, shapes, kw_chars)
        assert (__test_helper_native('overload_bind', native_impls, pin, shapes, kw_chars,
                                     kw_shapes)
                == rules.overload_bind(impls, pin, shapes, kw_chars, kw_shapes))
        answered += choice is not None
    assert answered > 1_000, answered


#: Definitions over every parameter kind, default spelling and annotation family
__test_helper_DEFINITIONS = '''
def plain(a, b, c=1, d=2.5): pass
def annotated(a: int, b: float = 1, c: int = 1.5, d: str | None = None, e: int = None): pass
def typeless(a: int = na, b: float = lib.na, c: Line = na, d: int = __dyn_default__,
             e: int = x.y.__dyn_default__, f: Optional[int] = None): pass
def kinds(a, /, b: int = 1, *args, c: float, d: int = 2, e='x', **kw): pass
def nested(a: list[int] = [], b: dict[str, float] = {}, c: 'int | None' = None,
           d: 'Pivot' = na, e: Union[int, None] = None, f: Any = None, g: NA[int] = None): pass
def only(*, a: int = 1, b: bool, c: Series[float] = close): pass
async def coro(a: int = 0, b: 'list[Pivot]' = None): pass
def empty(): pass
def positional_only(a: int, b: float = 1.0, /): pass
'''

#: Expressions for the dotted-name helpers and the typeless ``na`` spelling
__test_helper_EXPRESSIONS = [
    'na', '-na', '--lib.na', 'lib._na_none', '-(na)', 'x.na', '+na', 'na()', 'lib.na.x',
    '_na_none', '-_na_none', 'a.b.c', 'a().b', '(a).b', 'Series', 'lib.Persistent', 'x[1]',
    'NA', 'typing.NA', '1', 'not na', '-1', 'lib.NA',
]

#: Modules exercising the dataclass ``field`` bindings ``FactoryFields`` tracks
__test_helper_FACTORY_MODULES = ['''
from dataclasses import field
import dataclasses
import dataclasses as dc
from dataclasses import field as fld
@udt
class A:
    a: list = field(default_factory=lambda: [])
    b: list = dataclasses.field(default_factory=lambda: [])
    c: list = dc.field(default_factory=lambda: [])
    d: list = fld(default_factory=lambda: [])
    e: list = field(default_factory=lambda x: [])
    f: list = field(default_factory=list)
    g: list = field(1, default_factory=lambda: [])
    h: list = field(default_factory=lambda: [], init=False)
    i = field(default_factory=lambda: [])
    j: list = field(default=lambda: [])
@dataclass
class B:
    a: list = field(default_factory=lambda: [])
class C:
    a: list = field(default_factory=lambda: [])
@lib.udt
class D:
    a: list = dataclasses.field(default_factory=lambda: [])
field = 1
@udt
class E:
    a: list = field(default_factory=lambda: [])
def dataclasses(): pass
@udt
class F:
    b: list = dataclasses.field(default_factory=lambda: [])
if True:
    from dataclasses import field
    @udt
    class G:
        a: list = field(default_factory=lambda: [])
from .dataclasses import field
@udt
class H:
    a: list = field(default_factory=lambda: [])
from dataclasses import field
import dataclasses.x
@udt
class I:
    a: list = field(default_factory=lambda: [])
    b: list = dataclasses.field(default_factory=lambda: [])
@udt(frozen=True)
class J:
    a: list = field(default_factory=lambda: [])
''']


def __test_helper_ty_by_shape(node) -> str:
    """A deterministic ``ty_of`` that is not the stamp: the type follows the node's text."""
    return ('i', 'f', '?', 'o', 's', 'a:i')[len(ast.dump(node)) % 6]


def __test_helper_compare_definition(func, classes, ty_of, stats: Counter, where):
    unsupported = __test_helper_unsupported()
    try:
        native_sig = __test_helper_native('impl_sig', func, 'i', ty_of, dict(classes))
    except unsupported:
        stats['impl_sig unsupported'] += 1
    else:
        expected = rules.impl_sig(func, 'i', ty_of or rules.get_ty, classes)
        assert native_sig == tuple(expected), (where, func.name)
        stats['impl_sig'] += 1
    defaults = rules._param_defaults(func)  # noqa: internal API
    native_defaults = __test_helper_native('param_defaults', func)
    assert list(native_defaults) == list(defaults), (where, func.name)
    assert all(native_defaults[name] is defaults[name] for name in defaults), (where, func.name)
    args = func.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        default = defaults.get(arg.arg)
        try:
            got = __test_helper_native('param_fit', arg, default, ty_of, dict(classes))
        except unsupported:
            stats['param_fit unsupported'] += 1
            continue
        assert got == rules.param_fit(arg, default, ty_of or rules.get_ty, classes), \
            (where, func.name, arg.arg)
        stats['param_fit'] += 1
        if default is not None:
            assert (__test_helper_native('default_ty', default, ty_of)
                    == rules.default_ty(default, ty_of or rules.get_ty)), (where, arg.arg)
            stats['default_ty'] += 1


def __test_helper_corpus_trees():
    """The parity corpus of ``test_135`` as freshly parsed trees."""
    return [(path, ast.parse(path.read_text())) for path in __test_helper_inputs()]


def __test_native_definitions_match__():
    """The definition readers, ``FactoryFields`` and the dotted-name helpers, over the
    corpus as parsed and over hand-written forms."""
    __test_helper_require_native()
    stats = Counter()
    trees = [(Path('definitions'), ast.parse(__test_helper_DEFINITIONS))]
    trees += [(Path(f'factory{i}'), ast.parse(src)) for i, src in
              enumerate(__test_helper_FACTORY_MODULES)]
    trees += __test_helper_corpus_trees()
    for path, tree in trees:
        stamped = 0
        for node in ast.walk(tree):
            # A few stamps, so ``get_ty`` as ``ty_of`` reads something
            if isinstance(node, ast.expr) and stamped % 3 == 0:
                rules.set_ty(node, ('i', 'f', 'o:lib#Line')[stamped % 3 // 1 % 3])
            stamped += 1
        classes = {node.name: f'{path}#{node.name}' for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for class_map, ty_of in ((classes, None), ({}, __test_helper_ty_by_shape)):
                    __test_helper_compare_definition(node, class_map, ty_of, stats, path)
            if isinstance(node, ast.expr):
                assert __test_helper_native('dotted', node) == factory_fields.dotted_name(node)
                assert __test_helper_native('is_na_literal', node) == rules.is_na_literal(node)
                stats['expressions'] += 1
        class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        factories = factory_fields.FactoryFields(tree)
        native = __test_helper_native('factory_fields', tree, class_defs)
        for cls, calls in zip(class_defs, native, strict=True):
            expected = factories.of(cls)
            assert len(calls) == len(expected) and all(
                got is want for got, want in zip(calls, expected)), (path, cls.name)
            stats['factory classes'] += 1
            stats['factory calls'] += len(calls)
    for source in __test_helper_EXPRESSIONS:
        node = ast.parse(source, mode='eval').body
        assert __test_helper_native('dotted', node) == factory_fields.dotted_name(node), source
        assert __test_helper_native('is_na_literal', node) == rules.is_na_literal(node), source
        assert __test_helper_native('bare_wrapper', node) == rules.bare_wrapper(node), source
    print(dict(stats))
    assert stats['impl_sig'] > 500 and stats['factory calls'] >= 7, stats


#: Lowering-shaped expressions for ``stamp_lowering``
__test_helper_LOWERED = [
    'a + 1', '1 + 2.0', 'x if y else 1', '(q := 3) * 2', 'not z', '-1', '-na', 'f(1, k=2.5)',
    '[1, 2]', 'a < b < 1', 'a and b or 1', "'s' + 's'", 'a[1:2]', 'lambda: 1', '{1: 2, **d}',
    '(1).real', 'None', 'na + 1', 'x if 1 else 2.0', '1 // 2 % 3 ** 4', '1 << 2', "f'{x}'",
    '(a, b)', 'x[1][2].y(3)', '{a for a in b}', '-(1 + 2.5)', '~1', 'a if b else (c if d else 1)',
    '1 / 0 - 1.5', "b'x' + 1", '...', '[*a]', 'await x', '1 if x else "s"', '(1, 2) if x else (3, 4)',
]


def __test_helper_compare_lowering(root, result: str):
    """The native ``stamp_lowering`` over a copy, then the Python one over the tree."""
    native = __test_helper_native('stamp_lowering', root, result)
    rules.stamp_lowering(root, result)
    assert ast.dump(native) == ast.dump(root)
    assert ([getattr(node, rules.TY_ATTR, None) for node in ast.walk(native)]
            == [getattr(node, rules.TY_ATTR, None) for node in ast.walk(root)]), ast.dump(root)


def __test_native_stamp_lowering_matches__():
    """``stamp_lowering`` from scratch and over subtrees that are partly stamped."""
    __test_helper_require_native()
    rng = random.Random(1360)
    for source in __test_helper_LOWERED:
        for round_ in range(6):
            root = ast.parse(source, mode='eval').body
            if round_:
                for node in ast.walk(root):
                    if node is not root and isinstance(node, ast.expr) and rng.random() < 0.3:
                        rules.set_ty(node, rng.choice(__test_helper_PARAM_TYPES))
            __test_helper_compare_lowering(root, rng.choice(__test_helper_PARAM_TYPES))


#: The functions of two or more strings the pipeline's calls are recorded for
__test_helper_RECORDED = (
    'join', 'binop_type', 'unaryop_type', 'compare_type', 'head', 'array_of', 'matrix_of',
    'map_of', 'tuple_of', 'elements_of', 'class_of', 'element_of', 'key_of', 'value_of',
    'shape_mismatch', 'shape_conflict', 'render_ty', 'namespace_of', 'object_ty', 'class_id',
    'builtin_class_id', 'is_tuple', 'is_array', 'is_matrix', 'is_map', 'is_shaped', 'arity',
    'is_na_widen', 'pin_for', 'pin_chars', 'overload_result', 'default_fit', 'is_int_typed',
    'constant_type', 'overload_choice', 'overload_binding',
)

#: The functions over a tree the pipeline's calls are compared at the call
__test_helper_LIVE = (
    'annotation_type', 'annotation_takes_none', 'bare_wrapper', 'is_na_literal', 'dotted_name',
    'impl_sig', 'param_fit', 'default_ty', '_param_defaults', 'stamp_lowering',
)


def __test_helper_freeze(value):
    if isinstance(value, ast.operator | ast.unaryop):
        return type(value).__name__
    if isinstance(value, dict):
        return 'dict', tuple(value.items())
    if isinstance(value, list | tuple):
        return tuple(__test_helper_freeze(item) for item in value)
    return value


def __test_helper_compare_live(name, original, args, kwargs, state):
    """One call of a tree function, natively, before the Python one runs."""
    call = dict(zip(('node', 'second', 'third', 'fourth'), args)) | kwargs
    native = __test_helper_native
    if name == 'annotation_type':
        classes = dict(call.get('second', call.get('classes', {})))
        node = call['node']
        assert native(name, node, classes) == original(node, classes), ast.dump(node)
    elif name in ('annotation_takes_none', 'bare_wrapper', 'is_na_literal'):
        assert native(name, call['node']) == original(call['node'])
    elif name == 'dotted_name':
        assert native('dotted', call['node']) == original(call['node'])
    elif name == '_param_defaults':
        expected = original(call['node'])
        got = native('param_defaults', call['node'])
        assert list(got) == list(expected)
        assert all(got[key] is expected[key] for key in expected)
    elif name in ('impl_sig', 'param_fit'):
        ty_of = call.get('third', call.get('ty_of'))
        classes = dict(call.get('fourth', call.get('classes', {})))
        expected = original(call['node'], call['second'], ty_of or rules.get_ty, classes)
        expected = tuple(expected) if name == 'impl_sig' else expected
        assert native(name, call['node'], call['second'], ty_of, classes) == expected
    elif name == 'default_ty':
        ty_of = call.get('second', call.get('ty_of'))
        assert native(name, call['node'], ty_of) == original(call['node'], ty_of or rules.get_ty)
    elif name == 'stamp_lowering':
        root, result = call['node'], call['second']
        lowered = native(name, root, result)
        # The Python original runs right after this, on the tree itself, and
        # ``__test_helper_check_lowering`` compares its stamps then
        state.pending = (root, ast.dump(lowered),
                         [getattr(node, rules.TY_ATTR, None) for node in ast.walk(lowered)])
    state.stats[name] += 1


def __test_helper_check_lowering(state):
    if state.pending is None:
        return
    root, dumped, expected = state.pending
    state.pending = None
    assert ast.dump(root) == dumped
    assert [getattr(node, rules.TY_ATTR, None) for node in ast.walk(root)] == expected


def __test_helper_wrap(name, original, state):
    if name in __test_helper_RECORDED:
        def recorded(*args):
            state.calls[name].add(__test_helper_freeze(args))
            return original(*args)

        return recorded

    def live(*args, **kwargs):
        try:
            __test_helper_compare_live(name, original, args, kwargs, state)
        except __test_helper_unsupported():
            state.stats[f'{name} unsupported'] += 1
        out = original(*args, **kwargs)
        __test_helper_check_lowering(state)
        return out

    return live


def __test_helper_record(monkeypatch) -> SimpleNamespace:
    """Wrap the rules the pipeline calls: the pure ones are recorded, the ones over a
    tree compared at the call."""
    state = SimpleNamespace(calls={name: set() for name in __test_helper_RECORDED},
                            stats=Counter(), pending=None,
                            originals={name: getattr(rules, name) for name in
                                       __test_helper_RECORDED + __test_helper_LIVE})
    for module in list(sys.modules.values()):
        if not (getattr(module, '__name__', '') or '').startswith('pynecore.'):
            continue
        for name, original in state.originals.items():
            if getattr(module, name, None) is original:
                monkeypatch.setattr(module, name, __test_helper_wrap(name, original, state))
    return state


def __test_helper_replay(state):
    """Every recorded call of a pure function, natively."""
    for name, calls in state.calls.items():
        original = state.originals[name]
        for args in calls:
            if name in ('overload_choice', 'overload_binding'):
                impls = [rules.ImplSig(*impl) for impl in args[0]]
                rest = [dict(arg[1]) if isinstance(arg, tuple) and arg[:1] == ('dict',)
                        else arg for arg in args[1:]]
                choice = original(impls, *rest)
                native_rest = [list(arg) if isinstance(arg, tuple) else arg for arg in rest]
                got = __test_helper_native(name, [tuple(impl) for impl in impls], *native_rest)
                assert got == __test_helper_index(impls, choice), (name, args)
            elif name in ('binop_type', 'unaryop_type'):
                op = getattr(ast, args[0])()
                assert __test_helper_native(name, *args) == original(op, *args[1:]), args
            else:
                kind, expected = __test_helper_outcome(original, *args)
                if kind == 'raised':
                    state.stats[f'{name} (Python raises)'] += 1
                    continue
                native_args = args if name == 'constant_type' else [
                    list(arg) if isinstance(arg, tuple) else arg for arg in args]
                got = __test_helper_native(name, *native_args)
                assert __test_helper_plain(got) == __test_helper_plain(expected), (name, args)
            state.stats[f'{name} (replayed)'] += 1

def __test_helper_stamp_view(node, index: dict[int, int]):
    """Every stamp a node carries, node references as their walk position."""
    def canon(value):
        if isinstance(value, ast.AST):
            return 'node', index[id(value)]
        if isinstance(value, list | tuple):
            return type(value).__name__, tuple(canon(item) for item in value)
        if isinstance(value, dict):
            return 'dict', tuple((canon(k), canon(v)) for k, v in value.items())
        return value

    attrs = (rules.TY_ATTR, rules.PIN_ATTR, rules.PINS_ATTR, rules.VECTOR_ATTR,
             rules.VARYING_ATTR, rules.WIDEN_ATTR, rules.NA_CLASS_ATTR, rules.WIDEN_SITE_ATTR,
             rules.WIDENS_ATTR)
    return {attr: canon(node.__dict__[attr]) for attr in attrs if attr in node.__dict__}


#: The ``ast`` classes whose instances CPython's parser shares
__test_helper_SINGLETONS = (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


def __test_helper_compare_stamps(stmt, stats: Counter):
    """The accessors over one lowered statement: read natively, and written back."""
    nodes = list(ast.walk(stmt))
    index = {}
    for position, node in enumerate(nodes):
        index.setdefault(id(node), position)
    restamped = __test_helper_native('restamp', stmt)
    new_nodes = list(ast.walk(restamped))
    new_index = {}
    for position, node in enumerate(new_nodes):
        new_index.setdefault(id(node), position)
    assert len(new_nodes) == len(nodes)
    for node, new in zip(nodes, new_nodes):
        view = __test_helper_stamp_view(node, index)
        assert __test_helper_stamp_view(new, new_index) == view
        stats['restamped nodes'] += 1
        stats.update(f'stamp {attr}' for attr in view)
    for node in nodes:
        # The operator and context singletons are fields of the native tree, not nodes
        if isinstance(node, __test_helper_SINGLETONS) \
                or not __test_helper_stamp_view(node, index):
            continue
        native = __test_helper_native('stamps', node)
        expected = (rules.get_ty(node), rules.get_pin(node), rules.get_pins(node),
                    rules.get_vector(node), rules.get_varying(node), rules.get_widen(node),
                    rules.get_na_class(node), rules.get_widen_site(node), rules.get_widens(node))
        assert native == expected, ast.dump(node)
        stats['stamped nodes read'] += 1


def __test_native_stamp_accessors_match__():
    """Every accessor over every shape of value the Python setters write."""
    __test_helper_require_native()
    tree = ast.parse('def f(a):\n    x = g(a, 1.5) + h(na)\n    return k(x, -na)\n')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    func = tree.body[0]
    rules.set_varying(func, calls[:2])
    rules.set_ty(names[0], 'a:i')
    rules.set_pin(calls[0], 'i*')
    rules.set_pin(calls[1], None)
    rules.set_pins(calls[2], {0: 'if', 7: None, 2: '*I'})
    rules.set_pins(calls[1], None)
    rules.set_vector(calls[0], ((1, 'i', None), (), ('f', (2, ('x',)))))
    rules.set_vector(calls[2], None)
    rules.set_widen(names[1], rules.WIDEN_FLOAT)
    rules.set_widen(names[2], 'o:/Ádám/x.py#Pivot')
    rules.set_widen(names[3], None)
    rules.set_na_class(names[2], 'chr.SessionTimeRange')
    rules.set_widen_site(names[1], calls[1])
    rules.set_widen_site(names[3], None)
    rules.set_widens(calls[1], {3: rules.WIDEN_NUMBER_NA, 4: None})
    rules.set_widens(calls[0], None)
    rules.inherit_ty(names[4], names[0])
    rules.inherit_ty(names[3], calls[0])
    stats = Counter()
    __test_helper_compare_stamps(func, stats)
    for attr in ('_pine_ty', '_pine_pin', '_pine_pins', '_pine_vector', '_pine_varying',
                 '_pine_widen', '_pine_na_class', '_pine_widen_site', '_pine_widens'):
        assert stats[f'stamp {attr}'] >= 2 or attr in ('_pine_varying', '_pine_na_class'), attr


def __test_native_rules_over_the_pipeline_corpus__(monkeypatch):
    """Every rule call the Python pipeline makes over the parity corpus, natively too."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', 'python')
    recorder = __test_helper_record(monkeypatch)
    stats = Counter()
    unsupported = __test_helper_unsupported()
    modules = 0
    for path in __test_helper_inputs():
        text = path.read_text()
        tree = ast.parse(text)
        user, mode = import_hook._module_mode(tree)
        if not user:
            continue
        analysed = import_hook._analyse_tree(tree, text, path.resolve(), mode)
        # The stamps as the type pass leaves them, then as the lowering does
        for stmt in analysed.body:
            try:
                __test_helper_compare_stamps(stmt, stats)
            except unsupported:
                stats['statements not convertible'] += 1
        lowered, _ = import_hook._lower_tree(analysed, path.resolve(), mode)
        modules += 1
        for stmt in lowered.body:
            try:
                __test_helper_compare_stamps(stmt, stats)
            except unsupported:
                stats['statements not convertible'] += 1
    __test_helper_replay(recorder)
    stats.update(recorder.stats)
    print(f'{modules} modules', dict(sorted(stats.items())))
    assert modules > 10
    for name in ('annotation_type', 'impl_sig', 'stamp_lowering', 'join (replayed)',
                 'binop_type (replayed)', 'overload_choice (replayed)', 'param_fit'):
        assert stats[name] > 0, name
    assert not any('unsupported' in name for name in stats), stats
    assert 'statements not convertible' not in stats, stats
