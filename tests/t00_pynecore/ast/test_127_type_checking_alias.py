"""
The ``TYPE_CHECKING`` stripper and an aliased import of the flag.

The stripper removes ``TYPE_CHECKING`` from the ``typing`` import, so every
``if`` on the flag has to go with it -- including one spelled with the alias a
``from typing import TYPE_CHECKING as ...`` binds, or the module raises a
``NameError`` at import.
"""
import ast

from pynecore.transformers.type_checking_stripper import TypeCheckingStripperTransformer


def _strip(source: str) -> ast.Module:
    return TypeCheckingStripperTransformer().visit(ast.parse(source))


def _execute(tree: ast.Module) -> dict:
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(tree), '<type checking>', 'exec'), namespace)
    return namespace


def __test_aliased_flag_block_is_stripped__():
    """ An ``if`` on an alias of the flag is stripped together with the import """
    tree = _strip('from typing import TYPE_CHECKING as TC\n'
                  'if TC:\n'
                  '    import math\n'
                  'x = 1\n')
    assert ast.unparse(tree) == 'x = 1'
    assert _execute(tree)['x'] == 1


def __test_aliased_flag_keeps_else_body__():
    """ What runs is the else body, as for the plain spelling """
    tree = _strip('from typing import TYPE_CHECKING as TC, Any\n'
                  'if TC:\n'
                  '    x = 1\n'
                  'else:\n'
                  '    x = 2\n')
    assert ast.unparse(tree) == 'from typing import Any\nx = 2'
    assert _execute(tree)['x'] == 2


def __test_alias_inside_function__():
    """ The alias is honored wherever the ``if`` stands """
    tree = _strip('from typing import TYPE_CHECKING as TC\n'
                  'def f():\n'
                  '    if TC:\n'
                  '        return 1\n'
                  '    return 2\n')
    assert _execute(tree)['f']() == 2


def __test_plain_forms_still_stripped__():
    """ ``TYPE_CHECKING`` and ``typing.TYPE_CHECKING`` keep working """
    tree = _strip('import typing\n'
                  'from typing import TYPE_CHECKING\n'
                  'if TYPE_CHECKING:\n'
                  '    x = 1\n'
                  'if typing.TYPE_CHECKING:\n'
                  '    y = 1\n'
                  'z = 3\n')
    assert ast.unparse(tree) == 'import typing\nz = 3'


def __test_other_reads_of_the_flag_are_false__():
    """ A read of the flag outside an ``if`` test outlives the import as False """
    tree = _strip('from typing import TYPE_CHECKING as TC\n'
                  'if not TC:\n'
                  '    x = 1\n'
                  'flag = TYPE_CHECKING if False else TC\n')
    namespace = _execute(tree)
    assert namespace['x'] == 1
    assert namespace['flag'] is False


def __test_rebound_flag_name_is_kept__():
    """ A name the module binds itself is not replaced """
    tree = _strip('from typing import TYPE_CHECKING\n'
                  'def f(TYPE_CHECKING):\n'
                  '    return TYPE_CHECKING\n')
    assert _execute(tree)['f'](5) == 5


def __test_module_without_flag_untouched__():
    """ A module that never names the flag comes back as it was """
    source = 'from typing import Any\nif x:\n    y = 1\n'
    tree = _strip(source)
    assert ast.unparse(tree) == ast.unparse(ast.parse(source))


def __test_parameter_shadowing_an_alias_keeps_the_if__():
    """ An ``if`` on a parameter that shadows an alias is a runtime branch """
    tree = _strip('from typing import TYPE_CHECKING as check\n'
                  'def f(check):\n'
                  '    if check:\n'
                  '        return 1\n'
                  '    return 2\n'
                  'flag = check\n')
    namespace = _execute(tree)
    assert namespace['f'](True) == 1
    assert namespace['f'](False) == 2
    assert namespace['flag'] is False


def __test_local_shadowing_leaves_global_reads_replaced__():
    """ A local binding of the flag name does not keep other scopes' reads alive """
    tree = _strip('from typing import TYPE_CHECKING as check\n'
                  'def f():\n'
                  '    check = 5\n'
                  '    return check\n'
                  'def g():\n'
                  '    return check\n'
                  'h = lambda check=check: check\n'
                  'values = [check for check in (7,)]\n')
    namespace = _execute(tree)
    assert namespace['f']() == 5
    assert namespace['g']() is False
    assert namespace['h']() is False
    assert namespace['h'](9) == 9
    assert namespace['values'] == [7]


def __test_module_rebinding_keeps_the_import__():
    """ A flag name the module rebinds is not the flag: its import and its ``if`` stay """
    tree = _strip('from typing import TYPE_CHECKING as check\n'
                  'x = 1\n'
                  'if check:\n'
                  '    x = 2\n'
                  'check = True\n'
                  'if check:\n'
                  '    y = 3\n')
    namespace = _execute(tree)
    assert namespace['x'] == 1
    assert namespace['y'] == 3


def __test_nested_guards_in_else_bodies__():
    """ A guard nested in a guard's else body splices into a flat statement list """
    tree = _strip('from typing import TYPE_CHECKING\n'
                  'if TYPE_CHECKING:\n'
                  '    x = 1\n'
                  'else:\n'
                  '    if TYPE_CHECKING:\n'
                  '        x = 2\n'
                  '    else:\n'
                  '        x = 3\n'
                  '        y = 4\n'
                  'def f():\n'
                  '    if TYPE_CHECKING:\n'
                  '        return 1\n'
                  '    else:\n'
                  '        if TYPE_CHECKING:\n'
                  '            return 2\n'
                  '    return 5\n')
    assert ast.unparse(tree).startswith('x = 3\ny = 4\n')
    namespace = _execute(tree)
    assert namespace['y'] == 4
    assert namespace['f']() == 5


def __test_comprehension_target_reaches_nested_lambda__():
    """ A lambda in a comprehension reads the loop target, not the flag """
    tree = _strip('from typing import TYPE_CHECKING as check\n'
                  'values = [(lambda: check)() for check in (True, False)]\n')
    assert _execute(tree)['values'] == [True, False]


def __test_class_local_does_not_reach_comprehension__():
    """ A class-local binding does not shadow the flag inside a comprehension in the class """
    tree = _strip('from typing import TYPE_CHECKING as check\n'
                  'class A:\n'
                  '    check = 1\n'
                  '    values = [check for _ in range(2)]\n')
    assert _execute(tree)['A'].values == [False, False]


def __test_nonlocal_keeps_enclosing_flag_import__():
    """ A nested ``nonlocal`` declaration keeps the enclosing function's flag import """
    tree = _strip('def outer():\n'
                  '    from typing import TYPE_CHECKING as check\n'
                  '    def inner():\n'
                  '        nonlocal check\n'
                  '        return check\n'
                  '    if check:\n'
                  '        return 1\n'
                  '    return inner()\n')
    assert _execute(tree)['outer']() is False


def __test_qualified_guard_on_typing_parameter_is_kept__():
    """ ``typing.TYPE_CHECKING`` on a parameter named ``typing`` is a runtime test """
    tree = _strip('import typing\n'
                  'from typing import TYPE_CHECKING\n'
                  'def f(typing):\n'
                  '    if typing.TYPE_CHECKING:\n'
                  '        return 1\n'
                  '    return 2\n'
                  'if typing.TYPE_CHECKING:\n'
                  '    x = 1\n')
    namespace = _execute(tree)
    assert 'x' not in namespace
    assert namespace['f'](type('Flag', (), {'TYPE_CHECKING': True})) == 1


def __test_qualified_guard_on_rebound_typing_is_kept__():
    """ A module that rebinds ``typing`` keeps every qualified guard """
    tree = _strip('import typing\n'
                  'typing = type("Flag", (), {"TYPE_CHECKING": True})\n'
                  'if typing.TYPE_CHECKING:\n'
                  '    x = 1\n'
                  'def f():\n'
                  '    if typing.TYPE_CHECKING:\n'
                  '        return 1\n'
                  '    return 2\n')
    namespace = _execute(tree)
    assert namespace['x'] == 1
    assert namespace['f']() == 1
