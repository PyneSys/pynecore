"""Regression test for persistent references in attribute / subscript assignment targets"""

import ast
from pynecore.transformers.persistent import PersistentTransformer


def _find_main(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main function not found")


def __test_persistent_attribute_assignment_target__():
    """Persistent base in an attribute assignment target is rewritten to the global name.

    ``persistent_udt.field = ...`` previously had only its value visited, leaving the base
    name un-transformed in the target. At runtime the bare name was no longer defined (the
    declaration had become the global) and raised ``NameError``. The transformer must visit
    the assignment targets too, so the base of an attribute or subscript target is rewritten.
    """
    test_code = '''
from pynecore.types import Persistent

def main():
    state: Persistent[Foo] = Foo()
    state.value = 5
    state.items[0] = 7
'''
    main_func = _find_main(PersistentTransformer().visit(ast.parse(test_code)))
    persistent_name = "__persistent_main·state__"

    attr_base: ast.expr | None = None
    subscript_base: ast.expr | None = None
    for stmt in ast.walk(main_func):
        if not isinstance(stmt, ast.Assign):
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Attribute) and target.attr == "value":
            attr_base = target.value  # base of `state.value = 5`
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
            subscript_base = target.value.value  # base of `state.items[0] = 7`

    assert isinstance(attr_base, ast.Name), "attribute target base is not a plain name"
    assert attr_base.id == persistent_name, \
        f"attribute target base should be {persistent_name}, got {attr_base.id}"
    assert isinstance(subscript_base, ast.Name), "subscript target base is not a plain name"
    assert subscript_base.id == persistent_name, \
        f"subscript target base should be {persistent_name}, got {subscript_base.id}"


def __test_non_persistent_attribute_target_unchanged__():
    """A non-persistent attribute assignment target keeps its original base name."""
    test_code = '''
def main():
    state = Foo()
    state.value = 5
'''
    main_func = _find_main(PersistentTransformer().visit(ast.parse(test_code)))

    attr_base: ast.expr | None = None
    for stmt in ast.walk(main_func):
        if not isinstance(stmt, ast.Assign):
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Attribute):
            attr_base = target.value

    assert isinstance(attr_base, ast.Name), "attribute target base is not a plain name"
    assert attr_base.id == "state", \
        f"non-persistent target base should stay 'state', got {attr_base.id}"


if __name__ == "__main__":
    __test_persistent_attribute_assignment_target__()
    __test_non_persistent_attribute_target_unchanged__()
