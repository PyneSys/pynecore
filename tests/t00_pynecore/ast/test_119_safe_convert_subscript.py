"""
A subscript index is a Python-native consumer of a Pine int: a list, a string or a
tuple refuses the float a Pine int travels as, so an index typed as a Pine int is
truncated at the subscript, while a series buffer read is left to the buffer.
"""
import ast

from pynecore.transformers.pine_type_rules import INT, STR, set_ty
from pynecore.transformers.safe_convert_transformer import SafeConvertTransformer


def _lowered(source: str, **names: str) -> str:
    """Lower ``source`` with the given names stamped; every literal, arithmetic
    expression and ``int()`` cast is stamped as a Pine int."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            set_ty(node, names[node.id])
        elif isinstance(node, ast.BinOp) or isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Name) and node.func.id == 'int':
            set_ty(node, INT)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            set_ty(node, INT)
    tree = SafeConvertTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def __test_an_int_typed_index_is_truncated__():
    """A Pine int index on a Python container is truncated natively"""
    assert "lst[safe_convert.native_int(n)]" in _lowered("x = lst[n]\n", n=INT)
    assert "lst[safe_convert.native_int(n + 1)]" in _lowered("x = lst[n + 1]\n", n=INT)
    # A store is the same consumer
    assert "lst[safe_convert.native_int(n)] = 1.0" in _lowered("lst[n] = 1.0\n", n=INT)


def __test_the_cast_itself_truncates_natively__():
    """``lst[int(x)]`` truncates once instead of casting to a Pine int first"""
    out = _lowered("x = lst[int(y)]\n")
    assert "lst[safe_convert.native_int(y)]" in out
    assert "safe_int" not in out
    # The cast inside a wider index expression keeps its Pine meaning
    assert "lst[safe_convert.native_int(safe_convert.safe_int(y) + 1)]" \
        in _lowered("x = lst[int(y) + 1]\n")


def __test_other_indexes_are_left_alone__():
    """Only an index typed as a Pine int is a candidate"""
    assert "d[key]" in _lowered("x = d[key]\n", key=STR)
    assert "d[key]" in _lowered("x = d[key]\n")
    assert "m[i, j]" in _lowered("x = m[i, j]\n", i=INT, j=INT)
    assert "lst[3]" in _lowered("x = lst[3]\n")


def __test_a_folded_int_literal_becomes_an_int__():
    """A constant folder leaves a Pine int literal as a float: truncate it in place"""
    assert "lst[2]" in _lowered("x = lst[2.0]\n")
    assert "'abc'[2]" in _lowered("x = 'abc'[2.0]\n")


def __test_slice_bounds_are_truncated__():
    """Every bound of a slice is an index"""
    out = _lowered("x = lst[a:b:c]\n", a=INT, b=INT, c=INT)
    assert "lst[safe_convert.native_int(a):safe_convert.native_int(b):safe_convert.native_int(c)]" in out
    assert "lst[:safe_convert.native_int(b)]" in _lowered("x = lst[:b]\n", b=INT)


def __test_a_series_buffer_read_is_left_to_the_buffer__():
    """The series buffer truncates in its own ``__getitem__``"""
    assert "__state__[3][n]" in _lowered("x = __state__[3][n]\n", n=INT)
    assert "__state·f__[0][n]" in _lowered("x = __state·f__[0][n]\n", n=INT)


def __test_a_range_counter_is_already_native__():
    """The counter of a ``range()`` loop needs no truncation inside the loop"""
    source = "for i in range(n):\n    x = lst[i]\ny = lst[i]\n"
    out = _lowered(source, i=INT, n=INT)
    assert "    x = lst[i]" in out
    assert "y = lst[safe_convert.native_int(i)]" in out
    # A module's own range() binds no native counter
    shadowed = "def range(a, b):\n    return a\nfor i in range(n, 2):\n    x = lst[i]\n"
    assert "lst[safe_convert.native_int(i)]" in _lowered(shadowed, i=INT, n=INT)
