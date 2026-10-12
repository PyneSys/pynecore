"""
The type pass marks every int that goes into a float-element container and every typeless
na that goes where a string goes, and ImplicitConversion converts them there; a container
argument of an overload group pins its element type.
"""
import ast
from typing import cast

import pytest

from pynecore.transformers.implicit_conversion import ImplicitConversionTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import (
    get_pin, get_varying, get_vector, get_widen, get_widen_site, get_widens,
)

__test_helper_PRELUDE = '''from pynecore import lib
from pynecore.types import Matrix
'''


def __test_helper_widen(body: str) -> tuple[str, list[str]]:
    """Infer and widen ``main``'s body; return the emitted body and the diagnostics."""
    source = __test_helper_PRELUDE + 'def main():\n' + ''.join(
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
    # anywhere else na stays the typeless na
    ('x: float = lib._na_none', 'x: float = lib._na_none'),
    ('w = 1.5 if lib.bar_index < 0 else lib._na_none',
     'w = 1.5 if lib.bar_index < 0 else lib._na_none'),
    ('a: list[float] = lib.array.new_float(0)\nlib.array.push(a, lib._na_none)',
     'lib.array.push(a, lib._na_none)'),
])
def __test_typeless_na_where_a_string_goes_is_empty__(body, expected):
    emitted, diags = __test_helper_widen(body)
    assert expected in emitted.splitlines()
    assert not diags


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
