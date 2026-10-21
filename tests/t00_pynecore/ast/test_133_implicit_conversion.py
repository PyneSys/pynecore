"""
The type pass marks every int that goes into a float-element container and every typeless
na that goes where a string goes, and ImplicitConversion converts them there; a container
argument of an overload group pins its element type.
"""
import ast
import sys
import types
from typing import cast

import pytest

from pynecore.transformers.implicit_conversion import ImplicitConversionTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    get_pin, get_ty, get_varying, get_vector, get_widen, get_widen_site, get_widens,
)

__test_helper_PRELUDE = '''from pynecore import lib
from pynecore.types import Matrix
'''

#: The prelude with a user type and a lib class in scope, the way a compiled
#: script binds them: a ``@udt`` at module level, ``Line`` imported from the types
__test_helper_UDT_PRELUDE = __test_helper_PRELUDE + '''from pynecore.core.pine_udt import udt
from pynecore.types import Line


@udt
class Pivot:
    price: float = 0.0
    idx: int = 0
    tag: str = ""


@udt
class Holder:
    top: Pivot = lib.na(Pivot)
'''


def __test_helper_widen(body: str, prelude: str = __test_helper_PRELUDE) -> tuple[str, list[str]]:
    """Infer and widen ``main``'s body; return the emitted body and the diagnostics."""
    source = prelude + 'def main():\n' + ''.join(
        f'    {line}\n' for line in body.strip().splitlines())
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    tree = cast(ast.Module, ImplicitConversionTransformer().visit(tree))
    emitted: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            emitted = [ast.unparse(stmt) for stmt in node.body]
    return '\n'.join(emitted), [d.message for d in table.diags]


@pytest.mark.parametrize("body,expected", [
    # array.from: an int next to a float goes in as a float; an all-int array stays int
    ('a = lib.array.from_items(1, 2.0)', 'a = lib.array.from_items(1.0, 2.0)'),
    ('a = lib.array.from_items(1, 2)', 'a = lib.array.from_items(1, 2)'),
    # one element put into a float container: a literal becomes a float literal, any
    # other int value is cast
    ('a: list[float] = lib.array.new_float(0)\nlib.array.push(a, 2)', 'lib.array.push(a, 2.0)'),
    ('a: list[float] = lib.array.new_float(0)\nn = lib.bar_index\nlib.array.set(a, 0, n)',
     'lib.array.set(a, 0, float(n))'),
    ('m: dict[str, float] = lib.map.new()\nlib.map.put(m, "k", 1)', "lib.map.put(m, 'k', 1.0)"),
    ('m: Matrix[float] = lib.matrix.new(1, 1, 0.0)\nlib.matrix.set(m, 0, 0, 5)',
     'lib.matrix.set(m, 0, 0, 5.0)'),
    # the same puts into an int container convert nothing
    ('a: list[int] = lib.array.new_int(0)\nlib.array.push(a, 2)', 'lib.array.push(a, 2)'),
    # elements copied in from an int container are converted as a copy
    ('a: list[float] = lib.array.new_float(0)\nb = lib.array.from_items(5, 6)\n'
     'lib.array.concat(a, b)', 'lib.array.concat(a, safe_convert.safe_float_items(b))'),
    ('m: Matrix[float] = lib.matrix.new(1, 1, 0.0)\nlib.matrix.add_row(m, 1, '
     'lib.array.from_items(8))', 'lib.matrix.add_row(m, 1, safe_convert.safe_float_items('
                                 'lib.array.from_items(8)))'),
    ('m: Matrix[float] = lib.matrix.new(1, 1, 0.0)\nk = lib.matrix.new(1, 1, 3)\n'
     'lib.matrix.concat(m, k)', 'lib.matrix.concat(m, safe_convert.safe_float_matrix(k))'),
])
def __test_ints_into_float_containers_are_converted__(body, expected):
    emitted, diags = __test_helper_widen(body)
    assert expected in emitted.splitlines()
    assert not diags


@pytest.mark.parametrize("body,expected", [
    # a typeless na where a string goes is the string na, the empty string
    ('s: str = lib._na_none', "s: str = ''"),
    ('s: str = "q"\ns = lib._na_none', "s = ''"),
    ('w = "up" if lib.bar_index < 0 else lib._na_none', "w = 'up' if lib.bar_index < 0 else ''"),
    ('a: list[str] = lib.array.new_string()\nlib.array.push(a, lib._na_none)',
     "lib.array.push(a, '')"),
    ('m: dict[str, str] = lib.map.new()\nlib.map.put(m, "k", lib._na_none)',
     "lib.map.put(m, 'k', '')"),
    ('a = lib.array.from_items("a", lib._na_none)', "a = lib.array.from_items('a', '')"),
    ('def tail(p: str) -> str:\n    return p\nx = tail(lib._na_none)', "x = tail('')"),
    ('def label() -> str:\n    return lib._na_none', "    return ''"),
    # one the function holds rather than spells is converted where it is the na
    ('def label(value=lib._na_none) -> str:\n    return value\nx = label()',
     "    return safe_convert.implicit_convert(value, 's')"),
])
def __test_typeless_na_where_a_string_goes_is_empty__(body, expected):
    emitted, diags = __test_helper_widen(body)
    assert expected in emitted.splitlines()
    assert not diags


@pytest.mark.parametrize("body,expected", [
    # a typeless na where a float or an int goes is the numeric na
    ('x: float = lib._na_none', 'x: float = lib._na_float'),
    ('x: float = 1.0\nx = lib._na_none', 'x = lib._na_float'),
    ('i: int = lib._na_none', 'i: int = lib._na_float'),
    ('w = 1.5 if lib.bar_index < 0 else lib._na_none',
     'w = 1.5 if lib.bar_index < 0 else lib._na_float'),
    ('w = lib.close * 2.0 if lib.bar_index < 0 else lib._na_none * lib.close',
     'w = lib.close * 2.0 if lib.bar_index < 0 else lib._na_float * lib.close'),
    ('a: list[float] = lib.array.new_float(0)\nlib.array.push(a, lib._na_none)',
     'lib.array.push(a, lib._na_float)'),
    ('a: list[int] = lib.array.new_int(0)\nlib.array.set(a, 0, lib._na_none)',
     'lib.array.set(a, 0, lib._na_float)'),
    ('m: dict[str, float] = lib.map.new()\nlib.map.put(m, "k", lib._na_none)',
     "lib.map.put(m, 'k', lib._na_float)"),
    ('m: dict[str, float] = lib.map.new()\n'
     'lib.map.put(m, "k", lib.close if lib.bar_index > 1 else lib._na_none)',
     "lib.map.put(m, 'k', lib.close if lib.bar_index > 1 else lib._na_float)"),
    ('def tail(p: float) -> float:\n    return p\nx = tail(lib._na_none)', 'x = tail(lib._na_float)'),
    ('def missing() -> float:\n    return lib._na_none', '    return lib._na_float'),
    ('def missing(p: int) -> int:\n    if p > 0:\n        return p\n    return lib._na_none',
     '    return lib._na_float'),
    # a signed na is spelled too: the na object's sign hands the na itself back
    ('e: int = -lib._na_none', 'e: int = lib._na_float'),
    ('def missing() -> float:\n    return -lib._na_none\nx = missing()', '    return lib._na_float'),
    # a typeless na held rather than spelled -- a defaulted parameter, a variable -- is
    # converted where it is the na, and any other value goes through unchanged
    ('def missing(value=lib._na_none) -> float:\n    return value\nx = missing()',
     "    return safe_convert.implicit_convert(value, 'n')"),
    ('def missing(value=lib._na_none) -> float:\n    return -value\nx = missing()',
     "    return safe_convert.implicit_convert(-value, 'n')"),
    ('def tail(p: float) -> float:\n    return p\ndef pass_on(v=lib._na_none):\n'
     '    return tail(v)\nx = pass_on()', "    return tail(safe_convert.implicit_convert(v, 'n'))"),
    ('def pick(v=lib._na_none):\n    return 1.5 if lib.bar_index < 0 else v\nx = pick()',
     "    return 1.5 if lib.bar_index < 0 else safe_convert.implicit_convert(v, 'n')"),
    ('v = -lib._na_none\nx: float = v', "x: float = safe_convert.implicit_convert(v, 'n')"),
    ('a: list[float] = lib.array.new_float(0)\nv = -lib._na_none\nlib.array.push(a, v)',
     "lib.array.push(a, safe_convert.implicit_convert(v, 'n'))"),
    # the dynamic-default marker is no na: the prologue gives the parameter its default
    ('def f(p: float = __dyn_default__) -> float:\n    if p is __dyn_default__:\n'
     '        p = lib.close\n    return p\nx = f()', 'def f(p: float=__dyn_default__) -> float:'),
    # the value a numeric container is built from or filled with
    ('a = lib.array.from_items(1.5, lib._na_none)', 'a = lib.array.from_items(1.5, lib._na_float)'),
    ('a = lib.array.new_float(3, lib._na_none)', 'a = lib.array.new_float(3, lib._na_float)'),
    ('a = lib.array.new_int(3, initial_value=lib._na_none)',
     'a = lib.array.new_int(3, initial_value=lib._na_float)'),
    ('m: Matrix[float] = lib.matrix.new(2, 2, lib._na_none)',
     'm: Matrix[float] = lib.matrix.new(2, 2, lib._na_float)'),
    ('a = lib.array.new_bool(3, lib._na_none)', 'a = lib.array.new_bool(3, lib._na_none)'),
    # a bool has no numeric na, and a NaN map key would never find itself again
    ('b: bool = lib._na_none', 'b: bool = lib._na_none'),
    ('def flag() -> bool:\n    return lib._na_none', '    return lib._na_none'),
    ('m: dict[float, str] = lib.map.new()\nlib.map.put(m, lib._na_none, "v")',
     "lib.map.put(m, lib._na_none, 'v')"),
    # a lib parameter keeps its na: there it is the absent value the lib tests for
    ('x = lib.nz(lib._na_none)', 'x = lib.nz(lib._na_none)'),
])
def __test_typeless_na_where_a_number_goes_is_the_numeric_na__(body, expected):
    emitted, diags = __test_helper_widen(body)
    assert expected in emitted.splitlines()
    assert not diags


@pytest.mark.parametrize("helper,first", [
    ('def missing() -> float:\n    return lib._na_none', 'lib._na_none'),
    ('def missing() -> float:\n    return lib._na_none', 'missing()'),
    ('def missing(value=lib._na_none) -> float:\n    return value', 'missing()'),
    ('def missing() -> float:\n    return -lib._na_none', 'missing()'),
])
def __test_a_numeric_na_map_dispatches_to_the_float_overload__(helper, first):
    """
    A float map whose first value is a typeless na reaches the float overload, whether
    the na is put in as the literal or returned by a function declared to return a float,
    spelled there (signed or not) or held
    """
    source = __test_helper_PRELUDE + '''from pynecore.core.overload import overload
import pynecore.lib.map

@overload
def f(m: dict[str, str]) -> str:
    return 'STR'

@overload
def f(m: dict[str, float]) -> str:
    return 'FLOAT'

''' + helper + '''

def main():
    m: dict[str, float] = lib.map.new()
    lib.map.put(m, "a", ''' + first + ''')
    lib.map.put(m, "b", 1.5)
    return f(m)
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    namespace: dict = {'__name__': 'numeric_na_map'}
    exec(compile(tree, 'test', 'exec'), namespace)
    assert namespace['main']() == 'FLOAT'


def __test_a_held_typeless_na_is_converted_and_any_other_value_kept__():
    """
    A defaulted parameter returned as a float goes out as the numeric na where it holds
    the typeless na, and as itself where a call site the analysis never saw passes a value
    """
    source = __test_helper_PRELUDE + '''
def missing(value=lib._na_none) -> float:
    return value

def label(value=lib._na_none) -> str:
    return value

def main():
    return missing(), label()
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    namespace: dict = {'__name__': 'held_typeless_na'}
    exec(compile(tree, 'test', 'exec'), namespace)
    number, text = namespace['main']()
    assert type(number) is float and number != number
    assert text == ''
    assert namespace['missing'](2) == 2 and type(namespace['missing'](2)) is int
    assert namespace['label']('x') == 'x'


@pytest.mark.parametrize("body,expected", [
    # a run-time string parameter of a lib function takes the string na
    ('l = lib.label.new(lib.bar_index, lib.close, lib._na_none, tooltip=lib._na_none)',
     "l = lib.label.new(lib.bar_index, lib.close, '', tooltip='')"),
    ('s = lib.array.join(lib.array.from_items("a"), lib._na_none)',
     "s = lib.array.join(lib.array.from_items('a'), '')"),
    ('t = lib.ticker.new(lib._na_none, lib._na_none)', "t = lib.ticker.new('', '')"),
    ('lib.log.info(lib._na_none)', "lib.log.info('')"),
    # compile-time text of the study takes the text na
    ('lib.plot(lib.close, title=lib._na_none)', "lib.plot(lib.close, title='na')"),
    ('lib.plotchar(lib.close > lib.open, lib._na_none, lib._na_none, text=lib._na_none)',
     "lib.plotchar(lib.close > lib.open, 'na', 'na', text='na')"),
    ('lib.hline(1.1, lib._na_none)', "lib.hline(1.1, 'na')"),
    ('lib.alertcondition(lib.close > 1, lib._na_none, message=lib._na_none)',
     "lib.alertcondition(lib.close > 1, 'na', message='na')"),
    ('p = lib.plot(lib.close)\nq = lib.plot(lib.open)\n'
     'lib.fill(p, q, lib.color.red, lib._na_none)', "lib.fill(p, q, lib.color.red, 'na')"),
    ('p = lib.plot(lib.close)\nq = lib.plot(lib.open)\n'
     'lib.fill(p, q, 1.0, 2.0, lib.color.red, lib.color.blue, lib._na_none)',
     "lib.fill(p, q, 1.0, 2.0, lib.color.red, lib.color.blue, 'na')"),
    # an na top_value followed by a color is the gradient form: its bounds stay na
    ('p = lib.plot(lib.close)\nq = lib.plot(lib.open)\n'
     'lib.fill(p, q, lib._na_none, lib._na_none, lib.color.red, lib.color.blue, lib._na_none)',
     "lib.fill(p, q, lib._na_none, lib._na_none, lib.color.red, lib.color.blue, 'na')"),
    # an na top_color is no bool editable either: still the gradient form
    ('p = lib.plot(lib.close)\nq = lib.plot(lib.open)\n'
     'lib.fill(p, q, lib._na_none, 1.0, lib._na_none, lib.color.blue, lib._na_none)',
     "lib.fill(p, q, lib._na_none, 1.0, lib._na_none, lib.color.blue, 'na')"),
    # the hline form keeps its title third whatever follows
    ('h = lib.hline(1.0)\ng = lib.hline(2.0)\n'
     'lib.fill(h, g, lib._na_none, lib._na_none, lib._na_none)',
     "lib.fill(h, g, lib._na_none, 'na', lib._na_none)"),
    # an na color followed by a bool editable is the solid form
    ('p = lib.plot(lib.close)\nq = lib.plot(lib.open)\n'
     'lib.fill(p, q, lib._na_none, lib._na_none, True)',
     "lib.fill(p, q, lib._na_none, 'na', True)"),
    # a parameter that is no Pine string keeps its na
    ('m = lib.matrix.new(1, 1, 0.0)\nlib.matrix.sort(m, 0, lib._na_none)',
     'lib.matrix.sort(m, 0, lib._na_none)'),
    ('x = lib.nz(lib._na_none)', 'x = lib.nz(lib._na_none)'),
])
def __test_typeless_na_lib_string_argument__(body, expected):
    emitted, diags = __test_helper_widen(body)
    assert expected in emitted.splitlines()
    assert not diags


def __test_typeless_na_builtin_method_string_argument__():
    """The method spelling of a builtin binds its arguments the way the direct one does"""
    source = __test_helper_PRELUDE + '''from pynecore.core.pine_method import method_call

def main():
    l = lib.label.new(lib.bar_index, lib.close, "x")
    method_call('set_text', l, lib._na_none)
    a = lib.array.from_items("a", "b")
    s = method_call('join', a, lib._na_none)
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    tree = cast(ast.Module, ImplicitConversionTransformer().visit(tree))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')
    emitted = [ast.unparse(stmt) for stmt in main.body]
    assert "method_call('set_text', l, '')" in emitted
    assert "s = method_call('join', a, '')" in emitted
    assert not table.diags


def __test_a_declared_container_is_the_constructors_template__():
    """``Matrix[float]`` built from an int initial value builds floats, with no mismatch"""
    emitted, diags = __test_helper_widen('''
m: Matrix[float] = lib.matrix.new(1, 1, -3)
a: list[float] = lib.array.new(2, 1)
m = lib.matrix.new(2, 2, 7)
''')
    assert emitted.splitlines() == [
        'm: Matrix[float] = lib.matrix.new(1, 1, -3.0)',
        'a: list[float] = lib.array.new(2, 1.0)',
        'm = lib.matrix.new(2, 2, 7.0)',
    ]
    assert not diags


def __test_a_conversion_that_differs_per_context_is_per_instance__():
    """
    A body putting into a float array in one context and an int array in another
    converts per instance: the value becomes an instance-varying site whose vector
    entry is the conversion, each call site hands its instance its own, and nothing is
    reported. A context putting a float in has no say.
    """
    source = __test_helper_PRELUDE + '''
def put(a, v):
    lib.array.push(a, v)

def put_float(a, v):
    lib.array.push(a, v)

def make(v):
    a: list[float] = lib.array.from_items(v)
    return a

def main():
    put(lib.array.new_float(0), 1)
    put(lib.array.new_int(0), 1)
    put_float(lib.array.new_float(0), 1)
    put_float(lib.array.new_float(0), 2.0)
    make(1)
    make(2.0)
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    defs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = {name: next(call for call in ast.walk(node) if isinstance(call, ast.Call))
             for name, node in defs.items() if name != 'main'}
    value = calls['put'].args[1]
    assert get_widen(value) is None
    site = get_widen_site(value)
    assert site is not None
    assert sorted(set((get_widens(site) or {}).values()), key=str) == [None, 'f']
    assert get_varying(defs['put']) == [site]
    assert get_widen(calls['put_float'].args[1]) == 'f'
    assert get_widen_site(calls['put_float'].args[1]) is None
    assert get_widen(calls['make'].args[0]) == 'f'
    assert sorted((get_vector(stmt.value) for stmt in defs['main'].body[:2]
                   if isinstance(stmt, ast.Expr)), key=str) == [('f',), None]
    reasons = [d.origin.reason for d in table.diags if d.origin is not None]
    assert 'context-dependent-widen' not in reasons

    tree = cast(ast.Module, ImplicitConversionTransformer().visit(tree))
    assert ast.unparse(defs['put'].body[0]) == \
        'lib.array.push(a, safe_convert.implicit_convert(v, None))'
    assert any(isinstance(stmt, ast.ImportFrom) and stmt.module == 'pynecore.core'
               for stmt in tree.body)


def __test_a_method_call_hands_its_instance_the_conversion__():
    """
    A method passed to ``method_call`` as a function is bound by the call site's dispatch
    entry, so its call site carries the vector of the instance it creates, like a direct
    call
    """
    source = __test_helper_PRELUDE + '''from pynecore.core.pine_method import method_call

def put(a, v):
    lib.array.push(a, v)

def main():
    method_call(put, lib.array.new_float(0), 1)
    put(lib.array.new_int(0), 1)
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'main')
    assert [get_vector(cast(ast.Expr, stmt).value) for stmt in main.body] == [('f',), None]
    reasons = [d.origin.reason for d in table.diags if d.origin is not None]
    assert 'context-dependent-widen' not in reasons


def __test_a_conversion_no_instance_can_be_told_is_reported__():
    """
    A module whose pins are taken away (a failed container merge) has no instance
    vectors, so a conversion that differs per context is dropped and reported
    """
    source = __test_helper_PRELUDE + '''
def put(a, v):
    lib.array.push(a, v)

def main():
    i: list[int] = lib.array.new_int(0)
    s: list[str] = lib.array.new_string(0)
    lib.array.concat(i, s)
    put(lib.array.new_float(0), 1)
    put(lib.array.new_int(0), 1)
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    put = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == 'put')
    value = cast(ast.Call, cast(ast.Expr, put.body[0]).value).args[1]
    assert get_widen_site(value) is None
    reasons = [d.origin.reason for d in table.diags if d.origin is not None]
    assert reasons.count('context-dependent-widen') == 1


def __test_a_class_na_that_differs_per_context_is_reported__():
    """
    A generic body whose typeless na goes out as a different class in each context
    has no vector entry for the na of a class, so the instances keep the typeless na
    and the drop is reported rather than silently reaching the wrong overload
    """
    source = __test_helper_UDT_PRELUDE + '''from pynecore.core.overload import overload


@udt
class Other:
    x: float = 0.0


@overload
def describe(o: Other) -> str:
    return 'OTHER'


@overload
def describe(p: Pivot) -> str:
    return 'PIVOT'


def pick(value, flag: bool):
    return value if flag else lib._na_none


def main(flag: bool):
    return describe(pick(Pivot(1.0), flag)), describe(pick(Other(1.0), flag))
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    pick = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'pick')
    returned = cast(ast.IfExp, cast(ast.Return, pick.body[0]).value)
    assert get_widen_site(returned.orelse) is not None
    reasons = [d.origin.reason for d in table.diags if d.origin is not None]
    assert reasons.count('context-dependent-widen') == 1


def __test_a_container_argument_pins_its_element_type__():
    """The pin of an overload site carries the element type of a scalar array or matrix"""
    source = __test_helper_PRELUDE + '''from pynecore.core.overload import overload

@overload
def f(a: list[int]) -> str:
    return 'INT'

@overload
def f(a: list[float]) -> str:
    return 'FLOAT'

@overload
def g(m: Matrix[int]) -> str:
    return 'INT'

@overload
def g(m: Matrix[float]) -> str:
    return 'FLOAT'

def main():
    i = f(lib.array.from_items(1, 2))
    x = f(lib.array.from_items(1, 2.0))
    e = f(lib.array.new_float(0))
    m = g(lib.matrix.new(1, 1, 0.5))
    k = g(lib.matrix.new(0, 0, lib.na(int)))
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    pins: dict[str, str | None] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        pins[target.id] = get_pin(stmt.value)
    assert pins == {'i': 'I', 'x': 'F', 'e': 'F', 'm': '2', 'k': '1'}


def __test_a_container_pin_settles_the_overload_return_type__():
    """The implementation a container pin selects gives the call its return type"""
    source = __test_helper_PRELUDE + '''from pynecore.core.overload import overload

@overload
def f(a: list[int]) -> int:
    return 7 / 4

@overload
def f(a: list[float]) -> float:
    return 0.5

@overload
def g(x: int) -> str:
    return 'INT'

@overload
def g(x: float) -> str:
    return 'FLOAT'

def main():
    r = g(f(lib.array.new_int(0)))
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    outer = cast(ast.Call, cast(ast.Assign, main.body[0]).value)
    assert get_pin(outer.args[0]) == 'I'
    assert get_pin(outer) == 'i'


@pytest.mark.parametrize("body,expected", [
    # a ternary arm next to an object of a class is the na of that class
    ('w = Pivot(1.0) if lib.bar_index < 0 else lib._na_none',
     'w = Pivot(1.0) if lib.bar_index < 0 else lib.na(Pivot)'),
    # a declared variable, and a later store into a variable holding an object
    ('p: Pivot = lib._na_none', 'p: Pivot = lib.na(Pivot)'),
    ('p: Pivot = Pivot(1.0)\np = lib._na_none', 'p = lib.na(Pivot)'),
    # a field holding an object
    ('h = Holder(Pivot(1.0))\nh.top = lib._na_none', 'h.top = lib.na(Pivot)'),
    # an element of a container of objects: built with, pushed, declared
    ('a = lib.array.from_items(Pivot(1.0), lib._na_none)',
     'a = lib.array.from_items(Pivot(1.0), lib.na(Pivot))'),
    ('a: list[Pivot] = lib.array.new(0)\nlib.array.push(a, lib._na_none)',
     'lib.array.push(a, lib.na(Pivot))'),
    ('a: list[Pivot] = lib.array.from_items(lib._na_none)',
     'a: list[Pivot] = lib.array.from_items(lib.na(Pivot))'),
    # a parameter and a declared return
    ('def tail(p: Pivot) -> Pivot:\n    return p\nx = tail(lib._na_none)', 'x = tail(lib.na(Pivot))'),
    ('def missing() -> Pivot:\n    return lib._na_none', '    return lib.na(Pivot)'),
    # a constructor argument is the na of what the field holds: an object, a
    # number, a string
    ('h = Holder(lib._na_none)', 'h = Holder(lib.na(Pivot))'),
    ('p = Pivot(1.0, lib._na_none, lib._na_none)', "p = Pivot(1.0, lib._na_float, '')"),
    ('p = Pivot(lib._na_none)', 'p = Pivot(lib._na_float)'),
    ('p = Pivot(tag=lib._na_none)', "p = Pivot(tag='')"),
    # a lib class is spelled the way the module imports it
    ('x: Line = lib._na_none', 'x: Line = lib.na(Line)'),
    # a held typeless na is converted at run time, with the class
    ('def pick(v=lib._na_none):\n    return Pivot(1.0) if lib.bar_index < 0 else v\nx = pick()',
     '    return Pivot(1.0) if lib.bar_index < 0 else safe_convert.implicit_convert(v, Pivot)'),
    # a bool keeps the typeless na
    ('b: bool = lib._na_none', 'b: bool = lib._na_none'),
])
def __test_typeless_na_where_an_object_goes_is_the_na_of_its_class__(body, expected):
    emitted, diags = __test_helper_widen(body, __test_helper_UDT_PRELUDE)
    assert expected in emitted.splitlines()
    assert not diags


def __test_the_na_of_a_class_reaches_the_overload_of_that_class__():
    """
    An element put in as a typeless na next to an object reaches the implementation
    declaring the object's class, not the first one a typeless na would answer: the
    shape a library's ``to_string(timezone: Timezone)`` / ``to_string(range: Range)``
    group has, where the value dispatch took the enum overload for a typeless na
    """
    source = __test_helper_UDT_PRELUDE + '''from enum import StrEnum
from pynecore.core.overload import overload


class Zone(StrEnum):
    exchange = 'Exchange'


@overload
def describe(zone: Zone) -> str:
    return 'ZONE'


@overload
def describe(p: Pivot) -> str:
    return 'PIVOT'


def pick(flag: bool) -> Pivot:
    return Pivot(1.0) if flag else lib._na_none


def main(flag: bool):
    xs = lib.array.from_items(pick(flag))
    ys: list[Pivot] = lib.array.new(0)
    lib.array.push(ys, lib._na_none)
    return describe(lib.array.get(xs, 0)), describe(lib.array.get(ys, 0))
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    assert not table.diags, [d.message for d in table.diags]
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    namespace: dict = {'__name__': 'object_na_dispatch'}
    exec(compile(tree, 'test', 'exec'), namespace)
    assert namespace['main'](False) == ('PIVOT', 'PIVOT')
    assert namespace['main'](True) == ('PIVOT', 'PIVOT')
    assert namespace['pick'](False).type is namespace['Pivot']


def __test_a_held_typeless_na_becomes_the_na_of_the_class_it_goes_as__():
    """A defaulted parameter returned as an object goes out as the na of the class"""
    source = __test_helper_UDT_PRELUDE + '''
def missing(value=lib._na_none) -> Pivot:
    return value


def main():
    return missing()
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    namespace: dict = {'__name__': 'held_object_na'}
    exec(compile(tree, 'test', 'exec'), namespace)
    assert namespace['main']().type is namespace['Pivot']
    made = namespace['Pivot'](2.0)
    assert namespace['missing'](made) is made


def __test_a_field_default_naming_an_unbound_class_is_deferred__():
    """
    A field default runs while its class body does: the na of the class itself or of
    a class defined later is built for each object, once the class is bound
    """
    source = __test_helper_UDT_PRELUDE + '''

@udt
class Node:
    value: float = 0.0
    next: 'Node' = lib._na_none
    later: 'Later' = lib._na_none
    top: Pivot = lib._na_none


@udt
class Later:
    x: float = 0.0
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    assert not table.diags, [d.message for d in table.diags]
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    emitted = ast.unparse(tree).splitlines()
    assert 'from dataclasses import field as _dataclass_field' in emitted
    assert "    next: 'Node' = _dataclass_field(default_factory=lambda: lib.na(Node))" in emitted
    assert "    later: 'Later' = _dataclass_field(default_factory=lambda: lib.na(Later))" in emitted
    assert '    top: Pivot = lib.na(Pivot)' in emitted
    # The dataclass reads a string annotation's module, so the module is registered
    module = types.ModuleType('deferred_field_default')
    sys.modules[module.__name__] = module
    try:
        exec(compile(tree, 'test', 'exec'), module.__dict__)
        node = module.Node(1.0)
        assert node.next.type is module.Node
        assert node.later.type is module.Later
        assert node.top.type is module.Pivot
    finally:
        del sys.modules[module.__name__]


@pytest.mark.parametrize("definition", [
    # a parameter, a local bound after the na and a class-body field hide the class
    'def f(Pivot: int) -> Pivot:\n    return lib._na_none',
    'def f() -> Pivot:\n    p: Pivot = lib._na_none\n    Pivot = 3\n    return p',
    '@udt\nclass Shadow:\n    Pivot: int = 0\n    top: Pivot = lib._na_none',
])
def __test_a_class_whose_name_a_scope_binds_keeps_the_typeless_na__(definition):
    """The class's spelling would read the scope's own value, so nothing is converted"""
    tree = ast.parse(__test_helper_UDT_PRELUDE + definition + '\n')
    infer_module(tree, 'test')
    tree = cast(ast.Module, ImplicitConversionTransformer().visit(tree))
    emitted = ast.unparse(tree.body[-1])
    assert 'lib._na_none' in emitted
    assert 'lib.na(' not in emitted


#: The user-type prelude with a class holding a container of objects
__test_helper_BAG_PRELUDE = __test_helper_UDT_PRELUDE + '''

@udt
class Bag:
    items: list[Pivot] = lib.na(list)
'''


@pytest.mark.parametrize("body,expected", [
    # a declared variable, and a later store into a variable holding a container
    ('a: list[float] = lib._na_none', 'a: list[float] = lib.na(list)'),
    ('m: dict[str, float] = lib._na_none', 'm: dict[str, float] = lib.na(dict)'),
    ('x: Matrix[float] = lib._na_none', 'x: Matrix[float] = lib.na(Matrix)'),
    ('a: list[float] = lib.array.new_float(0)\na = lib._na_none', 'a = lib.na(list)'),
    # a ternary arm next to a container
    ('a = lib.array.new_float(0)\nb = a if lib.bar_index < 0 else lib._na_none',
     'b = a if lib.bar_index < 0 else lib.na(list)'),
    # a field holding a container: stored, and passed to the constructor
    ('b = Bag()\nb.items = lib._na_none', 'b.items = lib.na(list)'),
    ('b = Bag(lib._na_none)', 'b = Bag(lib.na(list))'),
    # a parameter and a declared return
    ('def size(a: list[float]) -> int:\n    return lib.array.size(a)\nn = size(lib._na_none)',
     'n = size(lib.na(list))'),
    ('def missing() -> dict[str, float]:\n    return lib._na_none', '    return lib.na(dict)'),
    # a held typeless na is converted at run time
    ('def pick(v=lib._na_none) -> list[float]:\n    return v\nx = pick()',
     "    return safe_convert.implicit_convert(v, 'A')"),
])
def __test_typeless_na_where_a_container_goes_is_the_na_of_that_container__(body, expected):
    emitted, diags = __test_helper_widen(body, __test_helper_BAG_PRELUDE)
    assert expected in emitted.splitlines()
    assert not diags


def __test_a_matrix_the_module_spells_nowhere_keeps_the_typeless_na__():
    """A matrix na is ``lib.na(<Matrix>)``, which needs the class the module imports"""
    emitted, diags = __test_helper_widen(
        'm = lib.matrix.new(1, 1, 0.0) if lib.bar_index < 0 else lib._na_none',
        'from pynecore import lib\n')
    assert emitted == 'm = lib.matrix.new(1, 1, 0.0) if lib.bar_index < 0 else lib._na_none'
    assert not diags


def __test_a_held_typeless_na_becomes_the_na_of_the_container_it_goes_as__():
    """A defaulted parameter returned as a container goes out as the na of the container"""
    source = __test_helper_PRELUDE + '''
def items(value=lib._na_none) -> list[float]:
    return value


def cells(value=lib._na_none) -> Matrix[float]:
    return value


def main():
    return items(), cells()
'''
    tree = ast.parse(source)
    infer_module(tree, 'test')
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    namespace: dict = {'__name__': 'held_container_na'}
    exec(compile(tree, 'test', 'exec'), namespace)
    array, matrix = namespace['main']()
    assert array.type is list
    assert matrix.type is namespace['Matrix']
    made = [1.0]
    assert namespace['items'](made) is made


def __test_a_typeless_na_handed_to_an_overload_group_is_the_na_its_choice_declares__():
    """
    A typeless na passed to a group is taken by the first implementation the other
    arguments let through, as at run time, and goes in as the na of what that
    implementation declares there -- an object, a container -- so the dispatch on
    the value selects the same one. A numeric position keeps the typeless na: the
    numeric na is a float the selector's exact pass would not take for an int
    """
    source = __test_helper_UDT_PRELUDE + '''from pynecore.core.overload import overload


@overload
def describe(p: Pivot, other: Pivot) -> str:
    return 'PIVOT'


@overload
def describe(h: Holder, items: list[Pivot]) -> str:
    return 'ITEMS'


@overload
def describe(h: Holder, name: str) -> str:
    return 'NAME'


@overload
def count(p: Pivot, n: int) -> str:
    return 'INT'


@overload
def count(p: Pivot, n: float) -> str:
    return 'FLOAT'


def main():
    h = Holder(Pivot(1.0))
    return describe(Pivot(1.0), lib._na_none), describe(h, lib._na_none), \\
        describe(h, name=lib._na_none), count(Pivot(1.0), lib._na_none)
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    assert not table.diags, [d.message for d in table.diags]
    tree = ast.fix_missing_locations(cast(ast.Module, ImplicitConversionTransformer().visit(tree)))
    emitted = ast.unparse(tree.body[-1])
    assert 'describe(Pivot(1.0), lib.na(Pivot))' in emitted
    assert 'describe(h, lib.na(list))' in emitted
    assert "describe(h, name='')" in emitted
    assert 'count(Pivot(1.0), lib._na_none)' in emitted
    namespace: dict = {'__name__': 'group_na'}
    exec(compile(tree, 'test', 'exec'), namespace)
    assert namespace['main']() == ('PIVOT', 'ITEMS', 'NAME', 'INT')


def __test_a_defaulted_implementation_the_walk_has_not_reached_is_selected__():
    """
    A body calls the group with fewer arguments than a LATER implementation
    declares: the call gets ahead of that definition, and the selection checks
    its defaults against the parameters the call omits, so they are typed before
    the body is read -- the shape of a library whose ``init`` implementations
    call each other
    """
    source = __test_helper_UDT_PRELUDE + '''from pynecore.core.overload import overload


@overload
def wrap(p: Pivot) -> Pivot:
    return wrap(Holder(p)).top


@overload
def wrap(box: Holder, flag: bool = True) -> Holder:
    return box


def main():
    return wrap(Pivot(1.0))
'''
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    assert not table.diags, [d.message for d in table.diags]
    inner = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node) == 'wrap(Holder(p))')
    assert get_ty(inner).endswith('#Holder')


@pytest.mark.parametrize("body,expected", [
    # the hand-written shape: a variable starts as a typeless na and a branch gives
    # it a value -- the na is the na of the type the assignments join to
    ('result = lib._na_none\nif lib.bar_index > 0:\n    result = lib.close\nx = result',
     'result = lib._na_float'),
    ('result = lib._na_none\nif lib.bar_index > 0:\n    result = 1\nelse:\n    result = 2.5',
     'result = lib._na_float'),
    ('label = lib._na_none\nif lib.bar_index > 0:\n    label = "up"', "label = ''"),
    ('top = lib._na_none\nif lib.bar_index > 0:\n    top = Pivot(1.0)', 'top = lib.na(Pivot)'),
    ('xs = lib._na_none\nif lib.bar_index > 0:\n    xs = lib.array.new_float(0)',
     'xs = lib.na(list)'),
    # a value the function computes decides it too
    ('def half(v):\n    return v / 2\nresult = lib._na_none\nif lib.bar_index > 0:\n'
     '    result = half(lib.close)', 'result = lib._na_float'),
    # a bool, and a name nothing else assigns, keep the typeless na
    ('flag = lib._na_none\nif lib.bar_index > 0:\n    flag = True', 'flag = lib._na_none'),
    ('nothing = lib._na_none\nx = nothing', 'nothing = lib._na_none'),
])
def __test_a_typeless_na_a_name_starts_as_is_the_na_of_what_it_is_assigned_later__(body,
                                                                                      expected):
    emitted, diags = __test_helper_widen(body, __test_helper_UDT_PRELUDE)
    assert expected in emitted.splitlines()
    assert not diags


def __test_a_typeless_na_a_name_starts_as_follows_the_context_it_is_assigned_in__():
    """
    The same body gives the name a float in one context and a string in another:
    the na it starts as is converted per instance, the way any other value is (the
    parameter's stamp, the join over both contexts, is what the report calls unknown)
    """
    body = ('def pick(v):\n    out = lib._na_none\n    if lib.bar_index > 0:\n        out = v\n'
            '    return out\na = pick(lib.close)\nb = pick("s")')
    emitted, _ = __test_helper_widen(body)
    assert '    out = safe_convert.implicit_convert(lib._na_none, None)' in emitted.splitlines()


@pytest.mark.parametrize("body,message", [
    ('x = 1.0\nif lib.bar_index > 0:\n    x = "s"',
     "'x' gets both float and string, which are different types"),
    ('result = lib._na_none\nif lib.bar_index > 0:\n    result = "s"\nelse:\n    result = 2',
     "'result' gets both string and int, which are different types"),
    ('x = True\nif lib.bar_index > 0:\n    x = Pivot(1.0)',
     "'x' gets both bool and Pivot, which are different types"),
])
def __test_branches_assigning_different_types_are_reported__(body, message):
    """Pine rejects a variable whose assignments disagree; the walk names the pair"""
    emitted, diags = __test_helper_widen(body, __test_helper_UDT_PRELUDE)
    assert message in diags


@pytest.mark.parametrize("body,expected", [
    # a tuple's start: each position is the na of the type its assignments join to
    ('pair = (lib._na_none, lib._na_none)\nif lib.bar_index > 0:\n    pair = (1, "a")\n'
     'else:\n    pair = (2.5, "b")\nn, s = pair', "pair = (lib._na_float, '')"),
    ('box = (lib._na_none, lib._na_none)\nif lib.bar_index > 0:\n'
     '    box = (Pivot(1.0), lib.array.new_float(0))\nx = box',
     'box = (lib.na(Pivot), lib.na(list))'),
    # a position a bool is assigned to keeps the typeless na
    ('mixed = (lib._na_none, lib._na_none)\nif lib.bar_index > 0:\n'
     '    mixed = (True, lib.close)\nx = mixed', 'mixed = (lib._na_none, lib._na_float)'),
])
def __test_a_tuple_a_name_starts_as_is_the_na_of_each_position_assigned_later__(body,
                                                                                 expected):
    emitted, diags = __test_helper_widen(body, __test_helper_UDT_PRELUDE)
    assert expected in emitted.splitlines()
    assert not diags


def __test_tuple_positions_assigned_different_types_are_reported_one_by_one__():
    """A conflict sits at a position, and every position that disagrees is named"""
    body = ('bad = (lib._na_none, lib._na_none)\nif lib.bar_index > 0:\n    bad = (1, "a")\n'
            'else:\n    bad = ("x", 2)')
    _, diags = __test_helper_widen(body)
    assert "element 1 of 'bad' gets both int and string, which are different types" in diags
    assert "element 2 of 'bad' gets both string and int, which are different types" in diags


@pytest.mark.parametrize("body,expected", [
    ("__sec_write__('sec-1', lib.close)\nvalue = __sec_read__('sec-1', lib._na_none)",
     "value = __sec_read__('sec-1', lib._na_float)"),
    ("__sec_write__('sec-2', (lib.high, 'tag', Pivot(1.0)))\n"
     "hi, tag, top = __sec_read__('sec-2', (lib._na_none, lib._na_none, lib._na_none))",
     "hi, tag, top = __sec_read__('sec-2', (lib._na_float, '', lib.na(Pivot)))"),
    # a bool position, and an id nothing writes, keep the typeless na
    ("__sec_write__('sec-3', (lib.close > lib.open, lib.close))\n"
     "up, c = __sec_read__('sec-3', (lib._na_none, lib._na_none))",
     "up, c = __sec_read__('sec-3', (lib._na_none, lib._na_float))"),
    ("value = __sec_read__('sec-4', lib._na_none)",
     "value = __sec_read__('sec-4', lib._na_none)"),
])
def __test_a_security_read_falls_back_to_the_na_of_what_its_write_publishes__(body,
                                                                               expected):
    """The default of the read the security transform emits is the write's na, per position"""
    emitted, _ = __test_helper_widen(body, __test_helper_UDT_PRELUDE)
    assert expected in emitted.splitlines()
