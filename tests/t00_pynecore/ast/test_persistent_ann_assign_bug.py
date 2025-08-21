"""Test case for persistent variable transformation in annotated assignments"""

import ast
from pynecore.transformers.persistent import PersistentTransformer


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

    # Parse and transform
    tree = ast.parse(test_code)
    transformer = PersistentTransformer()
    transformed_tree = transformer.visit(tree)

    # Find the main function
    main_func = None
    for node in transformed_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break

    assert main_func is not None, "main function not found"

    # Check that persistent variables are properly transformed
    persistent_name = "__persistent_main·length_2__"

    # Find alpha assignment
    alpha_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "alpha":
            alpha_assign = stmt
            break

    assert alpha_assign is not None, "alpha assignment not found"
    assert isinstance(alpha_assign.value, ast.Name), "alpha value should be a Name node"
    assert alpha_assign.value.id == persistent_name, \
        f"alpha value should be {persistent_name}, got {alpha_assign.value.id}"

    # Find beta assignment
    beta_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and \
                stmt.targets[0].id == "beta":  # noqa
            beta_assign = stmt
            break

    assert beta_assign is not None, "beta assignment not found"
    assert isinstance(beta_assign.value, ast.Name), "beta value should be a Name node"
    assert beta_assign.value.id == persistent_name, \
        f"beta value should be {persistent_name}, got {beta_assign.value.id}"

    # Find gamma assignment (complex expression)
    gamma_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "gamma":
            gamma_assign = stmt
            break

    assert gamma_assign is not None, "gamma assignment not found"
    assert isinstance(gamma_assign.value, ast.BinOp), "gamma value should be a BinOp node"
    assert isinstance(gamma_assign.value.left, ast.Name), "gamma left operand should be a Name node"
    assert gamma_assign.value.left.id == persistent_name, \
        f"gamma left operand should be {persistent_name}, got {gamma_assign.value.left.id}"

    print("✅ All tests passed!")


def __test_non_persistent_annotated_assignment__():
    """Test that non-persistent variables in annotated assignments are not transformed"""
    test_code = '''
def main():
    normal_var = 10
    alpha: float = normal_var  # This should NOT be transformed
    return alpha
'''

    # Parse and transform
    tree = ast.parse(test_code)
    transformer = PersistentTransformer()
    transformed_tree = transformer.visit(tree)

    # Find the main function
    main_func = None
    for node in transformed_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break

    assert main_func is not None, "main function not found"

    # Find alpha assignment
    alpha_assign = None
    for stmt in main_func.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "alpha":
            alpha_assign = stmt
            break

    assert alpha_assign is not None, "alpha assignment not found"
    assert isinstance(alpha_assign.value, ast.Name), "alpha value should be a Name node"
    assert alpha_assign.value.id == "normal_var", f"alpha value should be 'normal_var', got {alpha_assign.value.id}"

    print("✅ Non-persistent variable test passed!")


if __name__ == "__main__":
    __test_persistent_variable_in_annotated_assignment__()
    __test_non_persistent_annotated_assignment__()
