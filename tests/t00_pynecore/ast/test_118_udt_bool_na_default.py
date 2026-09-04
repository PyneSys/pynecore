"""
A UDT field's ``na(bool)`` default is lowered to a per-construction factory
bound to the canonical constructor: the spelling in effect at the class
statement decides, a later rebinding cannot reach into the default, and a
user's own ``na`` is not touched.
"""
import ast

from pynecore.transformers.dynamic_default import DynamicDefaultTransformer

FACTORY = "__pyne_field·__(default_factory=lambda: __pyne_bool_na·__())"


def _lower(source: str) -> str:
    return ast.unparse(DynamicDefaultTransformer().visit(ast.parse(source)))


def __test_lib_na_and_the_na_class_are_lowered__():
    lib_form = "@udt\nclass Flag:\n    f: bool = lib.na(bool)\n"
    assert FACTORY in _lower(lib_form)
    class_form = "from pynecore.types.na import NA as Na\n@udt\nclass Flag:\n    f: bool = Na(bool)\n"
    assert FACTORY in _lower(class_form)
    module_alias = "import pynecore.types.na as nt\n@udt\nclass Flag:\n    f: bool = nt.NA(bool)\n"
    assert FACTORY in _lower(module_alias)
    from_types = "from pynecore.types import na\n@udt\nclass Flag:\n    f: bool = na.NA(bool)\n"
    assert FACTORY in _lower(from_types)
    full_chain = ("import pynecore.types.na\n@udt\nclass Flag:\n"
                  "    f: bool = pynecore.types.na.NA(bool)\n")
    assert FACTORY in _lower(full_chain)
    other_package_import = ("import pynecore.types.na\nimport pynecore.core.script\n"
                            "@udt\nclass Flag:\n    f: bool = pynecore.types.na.NA(bool)\n")
    assert FACTORY in _lower(other_package_import)
    package_only = ("import pynecore.core.script\n@udt\nclass Flag:\n"
                    "    f: bool = pynecore.types.na.NA(bool)\n")
    assert FACTORY in _lower(package_only)
    aliased_then_root = ("import pynecore.types.na as nt\nimport pynecore\n@udt\nclass Flag:\n"
                         "    f: bool = pynecore.types.na.NA(bool)\n")
    assert FACTORY in _lower(aliased_then_root)
    package_alias = "import pynecore as p\n@udt\nclass Flag:\n    f: bool = p.types.na.NA(bool)\n"
    assert FACTORY in _lower(package_alias)
    types_alias = ("from pynecore import types as t\n@udt\nclass Flag:\n"
                   "    f: bool = t.na.NA(bool)\n")
    assert FACTORY in _lower(types_alias)
    other_constructor = ("import pynecore as p\n@udt\nclass Flag:\n"
                         "    f: bool = p.types.na.na_bool(bool)\n")
    assert FACTORY not in _lower(other_constructor)
    root_rebound = ("import pynecore.types.na\nimport pynecore.lib.math as pynecore\n"
                    "@udt\nclass Flag:\n    f: bool = pynecore.types.na.NA(bool)\n")
    assert FACTORY not in _lower(root_rebound)


def __test_the_binding_at_the_class_statement_decides__():
    """A rebinding after the class does not matter; one before it does"""
    after = ("from pynecore.types.na import NA\n@udt\nclass Flag:\n    f: bool = NA(bool)\n"
             "NA = lambda _: 'counterfeit'\n")
    assert FACTORY in _lower(after)
    before = ("from pynecore.types.na import NA\ndef NA(_):\n    return 0\n"
              "@udt\nclass Flag:\n    f: bool = NA(bool)\n")
    assert FACTORY not in _lower(before)
    own = "def na(_):\n    return 0\n@udt\nclass Flag:\n    f: bool = na(bool)\n"
    assert FACTORY not in _lower(own)
    rebound_module = ("import pynecore.types.na as nt\nimport pynecore.lib.math as nt\n"
                      "@udt\nclass Flag:\n    f: bool = nt.NA(bool)\n")
    assert FACTORY not in _lower(rebound_module)


def __test_only_a_udt_field_is_lowered__():
    plain = "class Stub(Protocol):\n    f: bool = lib.na(bool)\n"
    assert FACTORY not in _lower(plain)
    other_type = "@udt\nclass Box:\n    x: float = lib.na(float)\n"
    assert FACTORY not in _lower(other_type)


def __test_generated_imports_stay_behind_the_future_block__():
    source = ('"""doc"""\nfrom __future__ import annotations\n'
              "@udt\nclass Flag:\n    f: bool = lib.na(bool)\n")
    body = ast.parse(_lower(source)).body
    assert isinstance(body[0], ast.Expr)
    assert isinstance(body[1], ast.ImportFrom) and body[1].module == '__future__'
    assert {stmt.module for stmt in body[2:4] if isinstance(stmt, ast.ImportFrom)} \
        == {'dataclasses', 'pynecore.types.na'}
    compile(_lower(source), '<lowered>', 'exec')
