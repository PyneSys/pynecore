"""The type pass's ``PineTypeTable`` and its node numbering, native, against Python.

The native type pass will build the table in Rust and hand it to the Python passes
that read it (``function_isolation``, ``module_interface.build_interface``, the
import hook), so the bridge has to convert a table both ways without a trace: the
same classes, the same field values, the same container types in the same order.
"""
import ast
import copy
import dataclasses
import sys
from pathlib import Path

import pytest

from pynecore.core import import_hook
from pynecore.transformers import module_interface, pipeline
from pynecore.transformers.node_ids import ID_ATTR, assign_node_ids
from pynecore.transformers.pine_type_table import Binding, FuncSig, PineTypeTable

from .test_135_native_transform import __test_helper_inputs, __test_helper_require_native

#: A library publishing a class, a method on it and an overload group
__test_helper_SHAPES = '''"""
@pyne
"""
from pynecore.core.overload import overload
from pynecore.core.pine_method import method
from pynecore.core.pine_udt import udt

__all__ = ['Point', 'shift', 'scale', 'pick']


@udt
class Point:
    x: float = 0.0
    n: int = 0
    tag: str = ""


@method
def shift(self: Point, d: float) -> float:
    return self.x + d


def scale(v: float) -> float:
    return v * 2


@overload
def pick(x: int) -> int:
    return x


@overload
def pick(x: float) -> float:
    return x
'''

#: A script importing it: dependency records, an imported class, a method on that
#: class, and a spoiled container that takes every pin away
__test_helper_APP = '''"""
@pyne
"""
from pynecore.lib import array, math
from pynecore.core.pine_method import method
from ttab_shapes import Point, scale, pick


@method
def grow(self: Point, d: int) -> int:
    return self.n + d


def main(x: int):
    p = Point(1.0, 2, "a")
    y = grow(p, x)
    z = pick(y) + scale(1.5)
    ai = array.new_int(1, x)
    alias = ai
    array.concat(alias, ("x",))
    bad = array.last(ai)
    return math.max(bad, 1) + z
'''


def __test_helper_same(want: object, got: object, where: str) -> None:
    """Assert two values are equal down to the exact type of every container and
    value and the order of every ``list``, ``tuple`` and ``dict``."""
    assert type(got) is type(want), (where, type(want).__name__, type(got).__name__)
    if dataclasses.is_dataclass(want):
        for item in dataclasses.fields(want):
            __test_helper_same(getattr(want, item.name), getattr(got, item.name),
                               f'{where}.{item.name}')
    elif isinstance(want, dict):
        assert isinstance(got, dict) and len(got) == len(want), where
        for index, ((want_key, want_item), (got_key, got_item)) in enumerate(
                zip(want.items(), got.items())):
            __test_helper_same(want_key, got_key, f'{where}<key {index}>')
            __test_helper_same(want_item, got_item, f'{where}[{want_key!r}]')
    elif isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)) and len(got) == len(want), where
        for index, (want_item, got_item) in enumerate(zip(want, got)):
            __test_helper_same(want_item, got_item, f'{where}[{index}]')
    elif isinstance(want, frozenset):
        # A set has no order: the same members, each of its exact type
        assert got == want, where
        assert sorted(map(repr, got)) == sorted(map(repr, want)), where
    else:
        assert got == want, (where, want, got)


def __test_helper_check_table(table: PineTypeTable, where: str) -> None:
    """The table converted to its native form and back is the table."""
    from pynecore.transformers import _native

    back = _native._type_table(table)
    assert back is not table, where
    assert back == table, where
    __test_helper_same(table, back, where)


def __test_helper_check_node_ids(before: ast.Module, after: ast.Module, where: str) -> None:
    """The native numbering of the type pass's input is the one the pass stamped."""
    from pynecore.transformers import _native

    numbered, next_native = _native._node_ids(copy.deepcopy(before))
    assert next_native == assign_node_ids(copy.deepcopy(before)), where
    fieldless = (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)
    compared = 0
    for want, got in zip(ast.walk(after), ast.walk(numbered), strict=True):
        assert type(got) is type(want), where
        if not isinstance(want, fieldless):
            assert getattr(got, ID_ATTR) == getattr(want, ID_ATTR), (where, type(want).__name__)
            compared += 1
    assert compared > 0, where

    # A start other than 0 shifts every id
    shifted, next_shifted = _native._node_ids(copy.deepcopy(before), 1000)
    assert next_shifted == next_native + 1000, where
    assert getattr(shifted, ID_ATTR) == 1000, where


def __test_helper_analyse(items: list[tuple[Path, str]], monkeypatch) -> list[PineTypeTable]:
    """Run the analysing half of the pipeline over modules, checking the table and the
    node numbering of every type pass it runs (an imported module's included).

    :return: The tables the type pass built.
    """
    monkeypatch.setenv('PYNE_AST_BACKEND', 'python')
    tables: list[PineTypeTable] = []
    run = pipeline._run

    def checking(name, ctx, fn, *args):
        before = copy.deepcopy(args[0]) if name == 'PineType' else None
        tree = run(name, ctx, fn, *args)
        if before is not None:
            where = ctx.path.name
            __test_helper_check_node_ids(before, tree, where)
            __test_helper_check_table(tree._pine_types, where)
            tables.append(tree._pine_types)
        return tree

    monkeypatch.setattr(pipeline, '_run', checking)
    for path, text in items:
        tree = ast.parse(text)
        user, mode = import_hook._module_mode(tree)
        if user:
            import_hook._analyse_tree(tree, text, path.resolve(), mode)
    return tables


def __test_type_table_crosses_the_bridge__(monkeypatch):
    """Every table of the corpus, the lib modules included, converts both ways exactly,
    and the native ``assign_node_ids`` stamps the ids the type pass stamps."""
    __test_helper_require_native()
    items = [(path, path.read_text()) for path in __test_helper_inputs()]
    tables = __test_helper_analyse(items, monkeypatch)
    modules = sum(1 for _, text in items if import_hook._module_mode(ast.parse(text))[0])
    assert len(tables) == modules > 0


def __test_every_table_field_crosses_the_bridge__(tmp_path, monkeypatch):
    """A library and a script importing it fill every field of the table, so every
    conversion is checked on a value rather than on an empty container."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_SAVE_SCRIPT_TOML', '0')
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / 'ttab_shapes.py').write_text(__test_helper_SHAPES, encoding='utf-8')
    app = tmp_path / 'ttab_app.py'
    app.write_text(__test_helper_APP, encoding='utf-8')
    module_interface._registry.clear()
    module_interface._analysing.clear()
    try:
        tables = __test_helper_analyse([(app, __test_helper_APP)], monkeypatch)
    finally:
        module_interface._registry.clear()
        module_interface._analysing.clear()
        for name in [name for name in sys.modules if name.startswith('ttab_')]:
            del sys.modules[name]
    assert len(tables) == 2
    for item in dataclasses.fields(PineTypeTable):
        assert any(getattr(table, item.name) for table in tables), item.name
    nested: set[str] = set()

    def collect(value: object) -> None:
        nested.add(type(value).__name__)
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                collect(getattr(value, field.name))
        elif isinstance(value, dict):
            for key, element in value.items():
                collect(key)
                collect(element)
        elif isinstance(value, (list, tuple, frozenset)):
            for element in value:
                collect(element)

    for table in tables:
        collect(table)
    assert nested >= {'Unknown', 'Binding', 'FuncSig', 'ContextResult', 'CallSite', 'Diag',
                      'DepRecord', 'ExportSig', 'ClassSig', 'ImplSig'}


@pytest.mark.parametrize('table', [
    PineTypeTable(funcs={'f': FuncSig('f', params=('i',))}),
    PineTypeTable(bindings={'': {'x': Binding('x', line=True)}}),
    PineTypeTable(exportable={'f'}),
    PineTypeTable(call_pos={1: (2, 3, 4)}),
], ids=['tuple-for-list', 'bool-for-int', 'set-for-frozenset', 'long-position'])
def __test_a_table_of_another_shape_is_not_converted__(table):
    """A value of a type the field does not hold is no table the native side takes:
    the conversion declines rather than hand back something else."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    with pytest.raises(_native.Unsupported):
        _native._type_table(table)
