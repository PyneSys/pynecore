"""Immutable module constants retain Pine folding across function boundaries."""
import ast

import pytest

from pynecore.transformers.const_fold import ConstFoldTransformer
from pynecore.transformers.pine_type_rules import get_ty


def _fold(source: str) -> ast.Module:
    return ConstFoldTransformer().visit(ast.parse(source))


def _return(tree: ast.Module, name: str) -> ast.expr:
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == name)
    return next(node.value for node in function.body if isinstance(node, ast.Return))


def _execute(tree: ast.Module) -> dict:
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), '<constant folding>', 'exec'), namespace)
    return namespace


def __test_hoisted_weight_uses_pine_embedding_quantization__():
    tree = _fold('''from pynecore import lib
SW: float = 0.18
MW: float = 0.72
def main():
    return lib.math.max(0.0, 1.0 - SW - MW)
''')
    value = _return(tree, 'main')
    assert isinstance(value, ast.Constant)
    assert value.value == 0.1000000000000001
    assert get_ty(value) == 'f'


def __test_global_constant_chain_keeps_exact_intermediate_values__():
    tree = _fold('''SW = 0.18
MW = 0.72
WEIGHT = 1.0 - SW - MW
def main():
    return WEIGHT * 10.0
''')
    value = _return(tree, 'main')
    assert isinstance(value, ast.Constant)
    assert value.value == 1.0000000000000009


def __test_global_int_division_keeps_its_pine_type__():
    tree = _fold('''N: int = 14
D: int = 8
def main():
    return N / D
''')
    value = _return(tree, 'main')
    assert isinstance(value, ast.Constant)
    assert value.value == 1.75
    assert get_ty(value) == 'i'


def __test_nested_helpers_can_read_module_constants__():
    tree = _fold('''N = 14
def main():
    def helper():
        return N / 8
    return helper()
''')
    assert isinstance(_return(tree, 'helper'), ast.Constant)
    assert _execute(tree)['main']() == 1.75


@pytest.mark.parametrize('parameters,call', [
    ('N', 'main(6)'),
    ('N=6', 'main()'),
    ('N, /', 'main(6)'),
    ('*, N=6', 'main()'),
    ('*N', 'main(6)'),
    ('**N', 'main(value=6)'),
])
def __test_parameters_shadow_module_constants__(parameters, call):
    tree = _fold(f'N = 14\ndef main({parameters}):\n    return N\n')
    assert isinstance(_return(tree, 'main'), ast.Name)
    namespace = _execute(tree)
    assert eval(call, namespace) in (6, (6,), {'value': 6})


@pytest.mark.parametrize('binding', [
    'N = 6',
    'for N in (6,):\n        pass',
    'import math as N',
    'from math import pi as N',
    'def N():\n        return 6',
    'class N:\n        pass',
    'try:\n        pass\n    except Exception as N:\n        pass',
])
def __test_local_bindings_shadow_globals_for_the_whole_function__(binding):
    tree = _fold(f'N = 14\ndef main():\n    return N\n    {binding}\n')
    assert isinstance(_return(tree, 'main'), ast.Name)


def __test_outer_parameter_shadows_global_in_nested_helper__():
    tree = _fold('''N = 14
def main(N):
    def helper():
        return N / 2
    return helper()
''')
    assert isinstance(_return(tree, 'helper'), ast.BinOp)
    assert _execute(tree)['main'](6) == 3


def __test_shadowing_in_an_unrelated_function_keeps_global_foldable__():
    tree = _fold('''N = 14
def unrelated():
    N = 6
    return N
def main():
    return N / 8
''')
    assert isinstance(_return(tree, 'main'), ast.Constant)
    assert _execute(tree)['main']() == 1.75


@pytest.mark.parametrize('rebinding', [
    'N = 6',
    'N += 1',
    'del N',
    'if True:\n    N = 6',
    'import math as N',
])
def __test_rebound_module_values_are_not_captured__(rebinding):
    tree = _fold(f'N = 14\ndef main():\n    return N\n{rebinding}\n')
    assert isinstance(_return(tree, 'main'), ast.Name)


def __test_global_write_from_another_function_is_not_folded__():
    tree = _fold('''N = 14
def main():
    return N
def update():
    global N
    N = 6
''')
    assert isinstance(_return(tree, 'main'), ast.Name)
    namespace = _execute(tree)
    namespace['update']()
    assert namespace['main']() == 6


def __test_local_closure_values_are_not_treated_as_module_constants__():
    tree = _fold('''N = 14
def main():
    N = 6
    def helper():
        return N
    N = 8
    return helper()
''')
    assert isinstance(_return(tree, 'helper'), ast.Name)
    assert _execute(tree)['main']() == 8


@pytest.mark.parametrize('annotation', ['Persistent[int]', 'Series[int]', 'types.Persistent[int]'])
def __test_module_state_containers_are_not_captured__(annotation):
    tree = _fold(f'N: {annotation} = 14\ndef main():\n    return N\n')
    assert isinstance(_return(tree, 'main'), ast.Name)


def __test_class_locals_do_not_shadow_a_methods_module_global__():
    tree = _fold('''N = 14
class Container:
    N = 6
    def method(self):
        return N / 8
''')
    assert isinstance(_return(tree, 'method'), ast.Constant)
    assert _execute(tree)['Container']().method() == 1.75


def __test_async_functions_keep_module_constant_types__():
    tree = _fold('''N: int = 14
async def helper():
    return N / 8
''')
    value = _return(tree, 'helper')
    assert isinstance(value, ast.Constant)
    assert get_ty(value) == 'i'


def __test_wildcard_imports_prevent_module_constant_capture__():
    tree = _fold('''N = 14
from arbitrary_module import *
def main():
    return N
''')
    assert isinstance(_return(tree, 'main'), ast.Name)


def __test_pattern_capture_shadows_a_module_constant__():
    tree = _fold('''N = 14
def main(value):
    match value:
        case {"key": N}:
            pass
    return N
''')
    assert isinstance(_return(tree, 'main'), ast.Name)
    assert _execute(tree)['main']({'key': 6}) == 6
