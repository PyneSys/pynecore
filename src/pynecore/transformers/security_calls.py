"""
Recognition of the ``lib.request.security[_lower_tf](...)`` calls the security
passes lower.
"""
import ast

from . import ast_walk

#: Attribute names of the ``lib.request`` functions the security passes lower
_REQUEST_FUNCS = frozenset({'security', 'security_lower_tf'})


def has_security_call(tree: ast.AST) -> bool:
    """Whether ``tree`` holds a ``lib.request.security[_lower_tf](...)`` call.

    Every rewrite of the security passes starts at such a call, and most scripts
    have none: for them, this walk is all a security pass costs.

    :param tree: The tree to search.
    :return: True when at least one such call stands anywhere in it.
    """
    for node in ast_walk.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _REQUEST_FUNCS):
            continue
        owner = func.value
        if (isinstance(owner, ast.Attribute) and owner.attr == 'request'
                and isinstance(owner.value, ast.Name) and owner.value.id == 'lib'):
            return True
    return False


def is_security_call(node: ast.Call) -> bool:
    """Check if node is lib.request.security(...)."""
    return (isinstance(node.func, ast.Attribute)
            and node.func.attr == 'security'
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == 'request'
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == 'lib')


def is_security_lower_tf_call(node: ast.Call) -> bool:
    """Check if node is lib.request.security_lower_tf(...)."""
    return (isinstance(node.func, ast.Attribute)
            and node.func.attr == 'security_lower_tf'
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == 'request'
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == 'lib')
