"""Typed bool defaults are emitted without call-site anchors or lost type stamps."""
import ast

import pytest

from pynecore.core.import_hook import _analyse_tree, _lower_tree
from pynecore.transformers.pine_type_rules import BOOL, FLOAT, get_ty, elements_of


@pytest.mark.parametrize('expression, helper, expected', [
    ('close > open', '', BOOL),
    ('check(close)', 'def check(value):\n    return value > open\n', BOOL),
    ('pair()', 'def pair():\n    return close > open, close\n', 'tuple'),
])
def __test_inferred_bool_defaults_have_no_anchor__(tmp_path, expression, helper, expected):
    """Direct and helper-fed bools, including tuple returns, get a mode-aware default."""
    source = f'''"""@pyne"""
from pynecore.lib import close, open, request, script, syminfo
{helper}
@script.indicator("Test", na_bool=True)
def main():
    return request.security(syminfo.tickerid, "60", {expression})
'''
    path = tmp_path / 'typed_bool_default.py'
    tree = _analyse_tree(ast.parse(source), source, path, None)
    tree, layout = _lower_tree(tree, path, None, na_bool=True)
    read = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '__sec_read__')
    default = read.args[1]
    if expected == 'tuple':
        assert isinstance(default, ast.Tuple)
        assert elements_of(get_ty(default)) == (BOOL, FLOAT)
        default = default.elts[0]
    assert isinstance(default, ast.Call)
    assert isinstance(default.func, ast.Name)
    assert default.func.id == '__sec_bool_na·__'
    assert get_ty(default) == BOOL
    for node in ast.walk(default):
        if isinstance(node, ast.expr):
            assert hasattr(node, '_pine_ty')
    assert not any(slot.kind == 'anchor' for scope in layout.scopes.values() for slot in scope.slots)


@pytest.mark.parametrize('call, expected', [
    ('request.security(syminfo.tickerid, "60", close)', 'lib._na_none'),
    ('request.security_lower_tf(syminfo.tickerid, "1", close > open)', '[]'),
])
def __test_non_bool_and_lower_timeframe_defaults_are_preserved__(tmp_path, call, expected):
    """Numeric missing values and lower-timeframe empty arrays keep their existing contract."""
    source = f'''"""@pyne"""
from pynecore.lib import close, open, request, script, syminfo
@script.indicator("Test")
def main():
    return {call}
'''
    path = tmp_path / 'other_security_default.py'
    tree = _analyse_tree(ast.parse(source), source, path, None)
    tree, _ = _lower_tree(tree, path, None)
    read = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '__sec_read__')
    assert ast.unparse(read.args[1]) == expected
    assert '__sec_bool_na·__' not in ast.unparse(tree)
