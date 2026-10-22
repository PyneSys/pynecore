"""
Recognition of a script's entry point: the ``@script.indicator`` /
``@script.strategy`` / ``@script.library`` function the runner calls every bar.
"""
import ast

__all__ = ['is_script_entry']

_SCRIPT_ENTRY_DECORATORS = frozenset({'indicator', 'strategy', 'library'})


def is_script_entry(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the function is a ``@script.indicator/strategy/library`` entry.

    The runner calls such a function with NO arguments, so its parameter
    defaults are not fallbacks but the values it runs with, every bar. That is
    why the dynamic default pass leaves them alone -- the input machinery
    consumes them at ``def`` time -- and it is the same fact the type inference
    reads them by.

    :param node: The definition to inspect
    :return: True for a script entry point
    """
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if not (isinstance(target, ast.Attribute)
                and target.attr in _SCRIPT_ENTRY_DECORATORS):
            continue
        parent = target.value
        if isinstance(parent, ast.Name) and parent.id == 'script':
            return True
        if isinstance(parent, ast.Attribute) and parent.attr == 'script':
            return True
    return False
