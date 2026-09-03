"""
The int cast is lowered by the module kind: a script casts to a Pine int (a float),
a lib truncates to a native int, and a range() argument is always truncated.
"""
import ast

from pynecore.transformers.safe_convert_transformer import SafeConvertTransformer


def _lowered(source: str, *, lib: bool = False) -> str:
    tree = SafeConvertTransformer(lib=lib).visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def __test_a_lib_module_truncates_to_a_native_int__():
    """int() is Pine's cast in a script (a float) and the native truncation in a lib"""
    source = "x = int(y)\n"
    assert "safe_convert.safe_int(y)" in _lowered(source)
    assert "safe_convert.native_int(y)" in _lowered(source, lib=True)
    # float() is the same cast in both
    assert "safe_convert.safe_float(y)" in _lowered("x = float(y)\n", lib=True)


def __test_range_arguments_are_truncated__():
    """A range() argument is a Python-native consumer of a Pine int"""
    out = _lowered("for i in range(n, array.size(a), 2):\n    pass\n")
    assert "range(safe_convert.native_int(n), safe_convert.native_int(array.size(a)), 2)" in out
    # An int literal needs no truncation, and a keyword form is left alone
    assert "range(3)" in _lowered("for i in range(3):\n    pass\n")
    assert "from pynecore.core import safe_convert" in out


def __test_range_rewrite_leaves_starred_and_shadowed_forms_alone__():
    """A starred argument and a module's own range() are not the builtin consumer"""
    assert "range(*bounds)" in _lowered("for i in range(*bounds):\n    pass\n")
    shadowed = "def range(src, n):\n    return src\nfor i in range(a, 3):\n    pass\n"
    assert "native_int" not in _lowered(shadowed)
    imported = "from pynecore.lib.ta import range\nfor i in range(a, 3):\n    pass\n"
    assert "native_int" not in _lowered(imported)


def __test_the_module_name_binding_decides_the_import__():
    """Only a binding of the safe_convert name counts as the existing import"""
    out = _lowered("from pynecore.core.safe_convert import safe_int\nx = int(y)\n")
    assert out.count("safe_convert") == 3 and "from pynecore.core import safe_convert" in out
    relative = _lowered("from ..core import safe_convert\nx = int(y)\n", lib=True)
    assert relative.count("import") == 1
