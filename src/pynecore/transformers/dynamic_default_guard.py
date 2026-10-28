"""
Recognition of the parameter-default guards
:class:`~pynecore.transformers.dynamic_default.DynamicDefaultTransformer` emits.
"""
import ast

__all__ = ['SENTINEL_NAME', 'is_dynamic_default_guard']

#: The name a dynamic parameter default is replaced with; its guard compares the
#: parameter against it
SENTINEL_NAME = '__dyn_default__'


def is_dynamic_default_guard(stmt: ast.stmt) -> bool:
    """Whether a statement is one of the dynamic-default pass's parameter-default guards.

    A later pass that prepends its own prologue has to land AFTER these: the
    guard is what gives the parameter its value, so a statement placed above it
    reads the sentinel instead.

    :param stmt: A statement from a function body.
    :return: True if the statement resolves a dynamic parameter default.
    """
    if not (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
            and len(stmt.test.ops) == 1 and isinstance(stmt.test.ops[0], ast.Is)):
        return False
    right = stmt.test.comparators[0]
    return isinstance(right, ast.Name) and right.id == SENTINEL_NAME
