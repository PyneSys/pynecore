#!/usr/bin/env python3
"""
Generate ``pynecore/transformers/call_inline_templates.json`` from the lib sources.

``CallInlineTransformer`` replaces a call to a trivial Pine builtin with the
builtin's own body, written out as ONE expression. That expression is DERIVED
from the wrapper's source AST, never hand-written: the defining module is
parsed, the function body is accepted only in a restricted shape (docstring,
``if <test>: return <expr>`` guards, name aliases, one final ``return
<expr>``) and folded into a nested conditional expression. The same
operations run in the same order on the same operands, so the result is the
same double, the same ``na`` object and the same exception.

The derivation runs HERE, once, and the transform reads its result from the
JSON: the transform then reads no lib source and imports no lib module, so it
is the same function of the tree in every process (and in the native port,
which compiles the same file in). The lib sources are not part of the pipeline
hash and this file is, so an edit of an inlined body reaches cached script
bytecode only through the regenerated file -- and
``tests/t00_pynecore/ast/test_144_call_inline_templates.py`` fails, naming the
function whose template changed, until it is regenerated.

A body outside the shape is an error here, not a silent fallback: the
function has to leave :data:`INLINABLE_FUNCTIONS` explicitly.

``typing.cast`` is the one call folded away: it is defined to return its
second argument untouched, and the derivation verifies the wrapper's ``cast``
IS ``typing.cast`` before folding it.

``math.max`` / ``math.min`` are varargs and cannot come out of that shape.
They get one expansion written in the transform for a fixed positional
arity, guarded here by a structural check of their real bodies
(:func:`_varargs_shape`), so an edit to either body fails the generation
instead of changing what the expansion means.

The callee of a site must spell one of the paths recorded under
``callees`` once the module's import map resolved its base name: the
function's own module path, or a re-export of it by the packages a script
imports from. ``loop_ranges`` lists the spellings of ``pine_range`` the same
way: the counter of a ``for`` loop over it is never na.

Usage:
    python3 scripts/call_inline_collector.py
"""
import ast
import importlib
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any, cast

#: Registry format version. Bump whenever the shape below changes
SCHEMA_VERSION = 2

_SRC = Path(__file__).resolve().parent.parent / 'src'
JSON_PATH = _SRC / 'pynecore' / 'transformers' / 'call_inline_templates.json'

if not JSON_PATH.exists():
    # Importing pynecore transforms its own lib modules, and the transform reads
    # this registry: a missing file starts out empty (nothing is inlined) so the
    # import below can run and the real one be derived
    JSON_PATH.write_text(json.dumps({'v': SCHEMA_VERSION, 'functions': {}, 'callees': {},
                                     'loop_ranges': [], 'varargs_arities': []}) + '\n')

sys.path.insert(0, str(_SRC))

from pynecore.transformers.call_inline import SUPPORT_ALIAS_PREFIX  # noqa: E402

#: The functions whose body may be copied into the call site, and how the
#: expression is built: ``derive`` reads the body's own AST, ``varargs`` uses
#: the fixed-arity expansion of the transform. Every entry was read against its
#: body; a function missing from here simply keeps its call.
#:
#: Deliberately absent: ``math.avg`` / ``math.round`` / ``math.round_to_mintick``
#: (loops and multi-step arithmetic), ``array.set`` (a bare ``return`` no-op
#: branch is not an expression).
INLINABLE_FUNCTIONS: dict[tuple[str, str], str] = {
    ('pynecore.lib.math', 'abs'): 'derive',
    ('pynecore.lib.math', 'acos'): 'derive',
    ('pynecore.lib.math', 'asin'): 'derive',
    ('pynecore.lib.math', 'atan'): 'derive',
    ('pynecore.lib.math', 'ceil'): 'derive',
    ('pynecore.lib.math', 'cos'): 'derive',
    ('pynecore.lib.math', 'exp'): 'derive',
    ('pynecore.lib.math', 'floor'): 'derive',
    ('pynecore.lib.math', 'log'): 'derive',
    ('pynecore.lib.math', 'log10'): 'derive',
    ('pynecore.lib.math', 'pow'): 'derive',
    ('pynecore.lib.math', 'sign'): 'derive',
    ('pynecore.lib.math', 'sin'): 'derive',
    ('pynecore.lib.math', 'sqrt'): 'derive',
    ('pynecore.lib.math', 'tan'): 'derive',
    ('pynecore.lib.math', 'todegrees'): 'derive',
    ('pynecore.lib.math', 'toradians'): 'derive',
    ('pynecore.lib.math', 'max'): 'varargs',
    ('pynecore.lib.math', 'min'): 'varargs',
    ('pynecore.lib.array', 'get'): 'derive',
    ('pynecore.lib.array', 'size'): 'derive',
}

#: Free name of a copied body -> the attribute of :mod:`inline_support` that
#: anchors it. A name outside this table makes the body non-derivable, and the
#: pass's tests check every entry by object identity against the real module.
SUPPORT_NAMES: dict[str, dict[str, str]] = {
    'pynecore.lib.math': {
        'NA': 'NA', 'na_float': 'na_float', 'na_int': 'na_int',
        'builtins': 'py_builtins', 'math': 'py_math',
        'pine_math': 'pine_math', 'fdlibm': 'fdlibm',
        'float': 'py_float', 'int': 'py_int',
        'pine_int': 'py_float',
        '_na_of_operands': 'math_na_of_operands',
    },
    'pynecore.lib.array': {
        'NA': 'NA', 'na_float': 'na_float', 'na_int': 'na_int',
        'builtins': 'py_builtins',
        'float': 'py_float', 'int': 'py_int', 'len': 'py_len',
        'pine_int': 'py_float',
        '_na_element': 'array_na_element',
    },
}

#: Positional arities the varargs expansion is written for. Pine's own
#: ``math.max``/``math.min`` calls are two- or three-operand in practice;
#: anything wider keeps its call.
VARARGS_ARITIES = (2, 3)

#: Where ``pine_range`` is defined: the iterable whose loop counter is never na
LOOP_RANGE = ('pynecore.core.pine_range', 'pine_range')

#: The packages a script imports lib names from. A function one of them
#: re-exports is reachable under that spelling as well
REEXPORTING_PACKAGES = ('pynecore', 'pynecore.lib', 'pynecore.core')

#: Alias of an anchored support name
_SUPPORT_ALIAS = SUPPORT_ALIAS_PREFIX + '{}__'

#: Attribute the derivation tags the root of a subtree with that stands for an
#: actual argument (the transform's ``_ARG_MARK``)
_ARG_MARK = '_inl_argument'

#: Stand-in for the i-th parameter while a body is derived
_PARAM = '__inl_param{}__'


class NotDerivable(Exception):
    """A wrapper body does not have the shape the pass can copy."""


def _module_functions(module_name: str) -> dict[str, ast.FunctionDef]:
    """Parse a defining module and index its undecorated top-level defs.

    A decorated definition is skipped on purpose: the runtime name is the
    decorator's return value (an ``overload`` dispatcher), not the body below
    it, so copying that body would be copying the wrong function.

    :param module_name: Importable module name.
    :return: Function name -> its last undecorated definition.
    :raises NotDerivable: If the module has no readable source -- a pyc-only
        install, a zipapp or a frozen module.
    """
    try:
        module = importlib.import_module(module_name)
        source = Path(cast(str, module.__file__)).read_text(encoding='utf-8')
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise NotDerivable(f"{module_name} has no readable source: {exc!r}") from exc
    functions: dict[str, ast.FunctionDef] = {}
    for stmt in ast.parse(source).body:
        if isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list:
            functions[stmt.name] = stmt
    return functions


def _is_cast_call(node: ast.expr, module: types.ModuleType) -> bool:
    """Whether a node is ``cast(<type>, <value>)`` of ``typing.cast``.

    ``typing.cast`` is defined to return its second argument untouched, so
    folding it away removes a call and changes nothing else -- but only if the
    module's ``cast`` really is that function, which is checked here.

    :param node: The candidate node.
    :param module: The defining module object.
    :return: True when the call can be folded to its second argument.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'cast' and len(node.args) == 2 and not node.keywords):
        return False
    return getattr(module, 'cast', None) is typing.cast


def _copy_node(value: ast.AST) -> ast.AST:
    """Copy a node: its fields deeply, everything else hung on it as it is."""
    cls = value.__class__
    new = cls.__new__(cls)
    state = new.__dict__
    fields = cls._fields
    for field in fields:
        try:
            item = getattr(value, field)
        except AttributeError:
            continue
        if isinstance(item, ast.AST):
            item = _copy_node(item)
        elif isinstance(item, list):
            item = [_copy_node(entry) if isinstance(entry, ast.AST) else entry
                    for entry in item]
        state[field] = item
    for key, item in value.__dict__.items():
        if key not in fields:
            state[key] = item
    return new


def _arg_copy(node: ast.expr) -> ast.expr:
    """A detached copy of an argument expression, tagged as the user's own.

    :param node: The argument expression.
    :return: The tagged copy.
    """
    copied = cast(ast.expr, _copy_node(node))
    setattr(copied, _ARG_MARK, True)
    return copied


class _BodySubstitutor(ast.NodeTransformer):
    """Rewrite a copied body expression into call-site terms.

    Parameters and body-local aliases become the expressions bound for them,
    every other free name becomes its :mod:`inline_support` anchor, and
    ``typing.cast`` calls fold to their value argument.
    """

    def __init__(self, bindings: dict[str, ast.expr], supported: dict[str, str],
                 module: types.ModuleType):
        self.bindings = bindings
        self.supported = supported
        self.module = module

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if _is_cast_call(node, self.module):
            return self.visit(node.args[1])
        return cast(ast.expr, self.generic_visit(node))

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if not isinstance(node.ctx, ast.Load):
            raise NotDerivable(f"name {node.id!r} is written in the body")
        bound = self.bindings.get(node.id)
        if bound is not None:
            return _arg_copy(bound)
        anchor = self.supported.get(node.id)
        if anchor is None:
            raise NotDerivable(f"free name {node.id!r} has no inline_support anchor")
        return ast.Name(id=_SUPPORT_ALIAS.format(anchor), ctx=ast.Load())


def derive_expression(module_name: str, func_name: str,
                      args: list[ast.expr] | None = None) -> tuple[list[str], ast.expr]:
    """Reduce a wrapper's body to one conditional expression.

    Accepted shape, in order: an optional docstring, then any sequence of
    ``if <test>: return <expr>`` guards (no ``else`` branch needed) and
    ``<name> = <expr>`` aliases whose value folds to a name already in scope
    or to a constant, closed by a final ``return <expr>``.

    :param module_name: The defining module.
    :param func_name: The function name.
    :param args: One expression per parameter, the parameter names by default.
    :return: The parameter names and the expression, every argument read a
        tagged copy of its expression.
    :raises NotDerivable: If the signature or the body is outside the shape.
    """
    node = _module_functions(module_name).get(func_name)
    if node is None:
        raise NotDerivable(f"{module_name}.{func_name} has no undecorated definition")
    arguments = node.args
    if arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg or arguments.kwarg \
            or arguments.defaults:
        raise NotDerivable("only plain positional parameters are supported")
    params = [arg.arg for arg in arguments.args]
    module = importlib.import_module(module_name)
    supported = SUPPORT_NAMES.get(module_name, {})

    body = list(node.body)
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        body = body[1:]
    if not body:
        raise NotDerivable("empty body")

    if args is None:
        args = [ast.Name(id=param, ctx=ast.Load()) for param in params]
    bindings: dict[str, ast.expr] = dict(zip(params, args))

    def substitute(expr: ast.expr) -> ast.expr:
        return _BodySubstitutor(bindings, supported, module).visit(_copy_node(expr))

    def build(stmts: list[ast.stmt]) -> ast.expr:
        head = stmts[0]
        if isinstance(head, ast.Return):
            if len(stmts) != 1 or head.value is None:
                raise NotDerivable("a bare or non-final return is not an expression")
            return substitute(head.value)
        if isinstance(head, ast.Assign):
            if len(head.targets) != 1 or not isinstance(head.targets[0], ast.Name):
                raise NotDerivable("only a single-name alias is supported")
            folded = head.value
            while _is_cast_call(folded, module):
                folded = cast(ast.Call, folded).args[1]
            if isinstance(folded, ast.Name):
                if folded.id not in bindings:
                    raise NotDerivable(f"alias of {folded.id!r}, which is no parameter")
                bindings[head.targets[0].id] = bindings[folded.id]
            elif isinstance(folded, ast.Constant):
                bindings[head.targets[0].id] = folded
            else:
                raise NotDerivable("an alias must fold to a name or a constant")
            return build(stmts[1:])
        if isinstance(head, ast.If):
            if head.orelse or len(head.body) != 1 \
                    or not isinstance(head.body[0], ast.Return):
                raise NotDerivable("a guard must be a single `if <test>: return <expr>`")
            guarded = cast(ast.Return, head.body[0])
            if guarded.value is None or len(stmts) == 1:
                raise NotDerivable("a guard must return a value and be followed by more")
            return ast.IfExp(test=substitute(head.test),
                             body=substitute(guarded.value),
                             orelse=build(stmts[1:]))
        raise NotDerivable(f"{type(head).__name__} is not part of the derivable shape")

    return params, build(body)


def _varargs_shape(module_name: str, func_name: str) -> str:
    """Verify that ``math.max``/``math.min`` still have the body the expansion
    reproduces, and report which builtin they end in.

    Expected, exactly: a docstring, an ``assert``, a ``for n in numbers``
    loop whose only statement returns ``_na_of_operands(numbers)`` when ``n``
    is na, and a final ``return builtins.<max|min>(cast(..., numbers))``.

    :param module_name: The defining module.
    :param func_name: The function name.
    :return: The name of the builtin the body ends in.
    :raises NotDerivable: If the body drifted from that shape.
    """
    node = _module_functions(module_name).get(func_name)
    if node is None:
        raise NotDerivable(f"{module_name}.{func_name} has no undecorated definition")
    args = node.args
    if args.args or args.posonlyargs or args.kwonlyargs or args.kwarg or not args.vararg:
        raise NotDerivable("expected a pure varargs signature")
    vararg = args.vararg.arg
    body = list(node.body)
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        body = body[1:]
    if len(body) != 3 or not isinstance(body[0], ast.Assert):
        raise NotDerivable("expected `assert`, the na loop and the final return")
    loop, final = body[1], body[2]
    if not (isinstance(loop, ast.For) and not loop.orelse
            and isinstance(loop.target, ast.Name)
            and isinstance(loop.iter, ast.Name) and loop.iter.id == vararg
            and len(loop.body) == 1):
        raise NotDerivable("expected `for <n> in <numbers>:` with one statement")
    item = loop.target.id
    guard = loop.body[0]
    if not (isinstance(guard, ast.If) and not guard.orelse and len(guard.body) == 1
            and isinstance(guard.body[0], ast.Return)):
        raise NotDerivable("expected a single na guard in the loop")
    test = guard.test
    if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Compare)
            and len(test.operand.ops) == 1 and isinstance(test.operand.ops[0], ast.Eq)
            and isinstance(test.operand.left, ast.Name) and test.operand.left.id == item
            and isinstance(test.operand.comparators[0], ast.Name)
            and cast(ast.Name, test.operand.comparators[0]).id == item):
        raise NotDerivable("expected `not (<n> == <n>)` as the na test")
    returned = cast(ast.Return, guard.body[0]).value
    if not (isinstance(returned, ast.Call) and isinstance(returned.func, ast.Name)
            and returned.func.id == '_na_of_operands' and len(returned.args) == 1
            and isinstance(returned.args[0], ast.Name)
            and cast(ast.Name, returned.args[0]).id == vararg):
        raise NotDerivable("expected `_na_of_operands(<numbers>)` on the na path")
    if not isinstance(final, ast.Return):
        raise NotDerivable("expected a final return")
    call = final.value
    module = importlib.import_module(module_name)
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == 'builtins'
            and len(call.args) == 1):
        raise NotDerivable("expected a final `builtins.<name>(...)` call")
    inner = call.args[0]
    while _is_cast_call(inner, module):
        inner = cast(ast.Call, inner).args[1]
    if not (isinstance(inner, ast.Name) and inner.id == vararg):
        raise NotDerivable("expected the varargs tuple as the builtin's only argument")
    return call.func.attr


# --- encoding ----------------------------------------------------------------

#: The fields holding an operator or an expression context, by class name
_OPERATOR_FIELDS = frozenset({'op', 'ctx'})


def _encode_constant(value: Any) -> list[Any]:
    """A constant's value, tagged with its type: an int as its digits and a float
    as its ``repr``, so every reader parses them exactly.

    :param value: The value.
    :return: ``[type, value]``.
    :raises NotDerivable: For a value of another type, or a float without a
        finite spelling.
    """
    if value is None:
        return ['none', None]
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', str(value)]
    if type(value) is float:
        if value != value or value in (float('inf'), float('-inf')):
            raise NotDerivable(f"a float constant {value!r}")
        return ['float', repr(value)]
    if type(value) is str:
        return ['str', value]
    raise NotDerivable(f"a {type(value).__name__} constant")


def _encode(node: ast.AST, params: dict[str, int]) -> dict[str, Any]:
    """One node of a derived expression as JSON.

    A read of the i-th parameter is ``{"param": i}``; any other node is
    ``{"_type": <class>, <field>: <value>, ...}``, with ``"arg": true`` when it
    is the root of an argument stand-in. No location: the transform anchors its
    emission at the call site.

    :param node: The node.
    :param params: Parameter stand-in name -> its index.
    :return: The encoding.
    """
    state = node.__dict__
    if isinstance(node, ast.Name) and state.get(_ARG_MARK) and node.id in params:
        return {'param': params[node.id]}
    unknown = set(state) - set(node._fields) - set(node._attributes) - {_ARG_MARK}
    if unknown:
        raise NotDerivable(f"{type(node).__name__} carries {sorted(unknown)}")
    out: dict[str, Any] = {'_type': type(node).__name__}
    for field in node._fields:
        if field not in state:
            continue
        value = state[field]
        if isinstance(node, ast.Constant) and field == 'value':
            out[field] = _encode_constant(value)
        elif field in _OPERATOR_FIELDS:
            out[field] = type(value).__name__
        elif field == 'ops':
            out[field] = [type(op).__name__ for op in value]
        elif isinstance(value, ast.AST):
            out[field] = _encode(value, params)
        elif isinstance(value, list):
            out[field] = [_encode(item, params) if isinstance(item, ast.AST) else item
                          for item in value]
        else:
            out[field] = value
    if state.get(_ARG_MARK):
        out['arg'] = True
    return out


def _derived_entry(module_name: str, func_name: str) -> dict[str, Any]:
    """The registry entry of a ``derive`` function.

    :param module_name: The defining module.
    :param func_name: The function name.
    :return: Its parameters, the expression's source text and its encoding.
    """
    params, text = derive_expression(module_name, func_name)
    stand_ins = {_PARAM.format(index): index for index in range(len(params))}
    _, body = derive_expression(
        module_name, func_name,
        [ast.Name(id=name, ctx=ast.Load()) for name in stand_ins])
    return {'kind': 'derive', 'params': params, 'text': ast.unparse(text),
            'body': _encode(body, stand_ins)}


def _varargs_entry(module_name: str, func_name: str) -> dict[str, Any]:
    """The registry entry of a ``varargs`` function.

    :param module_name: The defining module.
    :param func_name: The function name.
    :return: The builtin the body ends in and the anchors the expansion reads.
    """
    supported = SUPPORT_NAMES[module_name]
    return {'kind': 'varargs', 'builtin': _varargs_shape(module_name, func_name),
            'na_anchor': supported['_na_of_operands'],
            'builtins_anchor': supported['builtins']}


def _spellings(target: tuple[str, str]) -> list[str]:
    """The dotted paths a script reaches a function under.

    :param target: ``(module, name)`` of the function.
    :return: Its own path, then every re-export by :data:`REEXPORTING_PACKAGES`.
    :raises NotDerivable: When the module does not define the function under its name.
    """
    module_name, name = target
    obj = getattr(importlib.import_module(module_name), name, None)
    if not isinstance(obj, types.FunctionType):
        raise NotDerivable(f"{module_name}.{name} is no function")
    out = [f'{module_name}.{name}']
    for package_name in REEXPORTING_PACKAGES:
        package = importlib.import_module(package_name)
        for attr, value in sorted(vars(package).items()):
            spelled = f'{package_name}.{attr}'
            if value is obj and spelled not in out:
                out.append(spelled)
    return out


def collect() -> dict[str, Any]:
    """Derive every template and spelling the transform reads.

    :return: The registry.
    :raises NotDerivable: If an allow-listed body is outside the shape.
    """
    functions: dict[str, Any] = {}
    callees: dict[str, str] = {}
    for (module_name, func_name), kind in sorted(INLINABLE_FUNCTIONS.items()):
        key = f'{module_name}.{func_name}'
        try:
            if kind == 'derive':
                functions[key] = _derived_entry(module_name, func_name)
            else:
                functions[key] = _varargs_entry(module_name, func_name)
            for spelled in _spellings((module_name, func_name)):
                callees[spelled] = key
        except NotDerivable as exc:
            raise NotDerivable(f"{key}: {exc}") from exc
    return {
        'v': SCHEMA_VERSION,
        'functions': functions,
        'callees': callees,
        'loop_ranges': _spellings(LOOP_RANGE),
        'varargs_arities': list(VARARGS_ARITIES),
    }


def write() -> None:
    """Collect the registry and write it next to the transformers."""
    registry = collect()
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write('\n')


if __name__ == '__main__':
    write()
