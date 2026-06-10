"""Test case for persistent variable transformation in annotated assignments"""

import ast

from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.slot_layout import ModuleLayout


def _find_main(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("main function not found")


def _is_state_ref(node: ast.expr | None, slot: int) -> bool:
    """Check that a node is a ``__state__[slot]`` reference."""
    return (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == "__state__"
            and isinstance(node.slice, ast.Constant) and node.slice.value == slot)


def __test_persistent_variable_in_annotated_assignment__():
    """Test that persistent variables are transformed in type-annotated assignments"""
    test_code = '''
from pynecore.types import Persistent

def main():
    length_2: Persistent[int] = 10
    alpha: float = length_2  # This should transform length_2
    beta = length_2  # This should also transform
    gamma: int = length_2 + 5  # Complex expression should also work
    return alpha + beta + gamma
'''
    main_func = _find_main(PersistentTransformer(ModuleLayout()).visit(ast.parse(test_code)))

    # Find alpha assignment
    alpha_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "alpha":
            alpha_assign = stmt
            break

    assert alpha_assign is not None, "alpha assignment not found"
    assert _is_state_ref(alpha_assign.value, 0), "alpha value should be a __state__[0] reference"

    # Find beta assignment
    beta_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and \
                stmt.targets[0].id == "beta":  # noqa
            beta_assign = stmt
            break

    assert beta_assign is not None, "beta assignment not found"
    assert _is_state_ref(beta_assign.value, 0), "beta value should be a __state__[0] reference"

    # Find gamma assignment (complex expression)
    gamma_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "gamma":
            gamma_assign = stmt
            break

    assert gamma_assign is not None, "gamma assignment not found"
    assert isinstance(gamma_assign.value, ast.BinOp), "gamma value should be a BinOp node"
    assert _is_state_ref(gamma_assign.value.left, 0), \
        "gamma left operand should be a __state__[0] reference"


def __test_non_persistent_annotated_assignment__():
    """Test that non-persistent variables in annotated assignments are not transformed"""
    test_code = '''
def main():
    normal_var = 10
    alpha: float = normal_var  # This should NOT be transformed
    return alpha
'''
    main_func = _find_main(PersistentTransformer(ModuleLayout()).visit(ast.parse(test_code)))

    # Find alpha assignment
    alpha_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "alpha":
            alpha_assign = stmt
            break

    assert alpha_assign is not None, "alpha assignment not found"
    assert isinstance(alpha_assign.value, ast.Name), "alpha value should be a Name node"
    assert alpha_assign.value.id == "normal_var", f"alpha value should be 'normal_var', got {alpha_assign.value.id}"


if __name__ == "__main__":
    __test_persistent_variable_in_annotated_assignment__()
    __test_non_persistent_annotated_assignment__()
