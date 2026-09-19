"""
Inline the body of trivial stateless Pine builtins into the call site.

A wrapper like ``math.abs`` is a few nanoseconds of work behind a Python
call that costs tens of them, and a script reaches such wrappers tens of
millions of times over a run (a rolling window loop calls ``math.max`` and
``array.get`` once per window element per bar). This pass replaces the call
with the wrapper's own body, written out as ONE expression::

    lib.math.abs(close - 1.0)
      ->  __inl·na_float__ if not ((__inl1·__ := lib.close - 1.0) == __inl1·__)
          else __inl·py_builtins__.abs(__inl1·__)

Bit-exactness by construction
-----------------------------

The emitted expression is DERIVED from the wrapper's own source AST, never
hand-written: the defining module is parsed, the function body is accepted
only in a restricted shape (docstring, ``if <test>: return <expr>`` guards,
name aliases, one final ``return <expr>``) and folded into a nested
conditional expression. The same operations run in the same order on the same
operands, so the result is the same double, the same ``na`` object and the
same exception. A body that grows past the shape stops being derivable and
the site keeps its call -- and the pass's tests fail on it, so the loss is
never silent.

``typing.cast`` is the one call folded away: it is defined to return its
second argument untouched, and the pass verifies the wrapper's ``cast`` IS
``typing.cast`` before folding it.

A guard a LITERAL argument decides is resolved at transform time
(``math.pow(x, 2)`` keeps only ``x * x``), and so are the operations that
guard reads: ``isinstance(2, NA)``, ``2 != 2``, ``int(0)``. Each is folded by
RUNNING the very same operation on that very literal, never by reasoning
about it, so an int-vs-float result cannot drift. A ``bool`` is not treated as
a numeric literal, so nothing folds on ``math.pow(x, true)``.

Arguments and evaluation order
------------------------------

An actual argument is classified by what re-evaluating it would cost: a
constant and a slot read the Series/Persistent lowering emitted
(``__state__[7]`` -- a list index with a compile-time-allocated index) run no
user code and cannot raise, a plain name read can only raise ``NameError``,
and anything else must run exactly once. A constant or a name goes in as
itself; a slot read is bound once only when the body reads it more than once;
anything else is bound once with an assignment expression, or goes in
directly when the body reads it exactly once on a path that always runs.

The call being replaced evaluated every argument, left to right, before
entering the body, while the copied body may read them in another order or
skip one entirely. When an impure argument is present and the body's natural
order over the raising arguments is not the source order, all of them are
forced in front as ``x is x`` probes, in source order -- a plain name is
probed without a temporary. With no impure argument nothing is forced: the
only difference reordering could make among names and slot reads is WHICH
``NameError`` a script with an undefined name reports, and the emission is not
worth an extra test per site for that.

Every comparison the pass emits for itself is marked ``pine_exact`` so the
later ``FloatToleranceTransformer`` leaves it alone -- but never a comparison
inside an ARGUMENT, which is the user's own Pine code and must still get the
tolerant rewrite.

``math.max`` / ``math.min`` are varargs and cannot come out of that shape.
They get one expansion written here for a fixed positional arity, guarded by
a structural check of their real bodies (:func:`_varargs_shape`), so an edit
to either body disables the expansion instead of changing what it means.

Name resolution
---------------

The copied body keeps its own module's names, which mean something else at
the call site: a script's ``math`` is ``pynecore.lib.math``, and
``BuiltinShadowTransformer`` lets it rebind ``abs`` or ``float``. Every free
name of a copied body therefore resolves through
:mod:`~pynecore.core.inline_support`, imported once per module under the same
collision-safe middle-dot alias the other passes use. A name that module does
not anchor makes the body non-derivable.

The callee itself must be PROVABLY the library function: the dotted path is
resolved through the module's import map at transform time and the resulting
object is compared by identity against the allow-listed function. A name the
module binds anywhere -- a user ``def abs``, a variable named ``math`` -- takes
the whole base name out of the pass.

Where a site is left alone
--------------------------

- module level, class bodies, decorators and default arguments: the emitted
  temporaries are assignment expressions, which would become globals or
  class attributes there;
- lambdas, comprehensions and generator expressions: an assignment expression
  is illegal in a comprehension's iterable and binds to the wrong scope
  elsewhere;
- keyword arguments, ``*args``, ``**kwargs``, or an arity the body does not
  take;
- a callee used as a value rather than called.

Leaving a call in place is always correct, so every unprovable case skips.

Placement in the pipeline
-------------------------

Runs in ``_lower_tree`` directly before ``FunctionIsolationTransformer``, so
no isolation wrapping is emitted for a site that no longer is a call, and
after the Series/Persistent passes, so the arguments it binds are already
lowered to slot reads. The type pass is finished by then; the inlined
expression is stamped with the call's own Pine type.

The wrapper bodies compare RAW (``x == x`` is their na test, ``exponent == 2``
a measured shortcut), which is why the pass marks its own comparisons exact
for ``FloatToleranceTransformer`` -- and only its own.
"""
from typing import Any, cast
import ast
import builtins
import importlib
import operator
import types
from pathlib import Path

from ..types.na import NA
from .pine_type_rules import get_ty, stamp_lowering
from .slot_layout import DEFAULT_STATE_PARAM

__all__ = ['CallInlineTransformer', 'INLINABLE_FUNCTIONS', 'SUPPORT_NAMES',
           'SUPPORT_MODULE', 'SUPPORT_ALIAS_PREFIX', 'derive_template',
           'varargs_expansion_arities']

#: The module every free name of an inlined body resolves through
SUPPORT_MODULE = 'pynecore.core.inline_support'

#: The package every inlinable function is defined in
_PACKAGE = 'pynecore'

#: Alias prefix/suffix of an anchored support name, and of a bound argument.
#: The middle dot cannot appear in a Python identifier, so neither a script
#: variable nor another pass's temporary can collide with these.
SUPPORT_ALIAS_PREFIX = '__inl·'
_SUPPORT_ALIAS = SUPPORT_ALIAS_PREFIX + '{}__'
_TEMP_NAME = '__inl{}·__'
#: Stand-in for an actual argument while the emission's shape is decided
_PLACEHOLDER = '__inlp{}·__'

#: The functions whose body may be copied into the call site, and how the
#: expression is built: ``derive`` reads the body's own AST, ``varargs`` uses
#: the fixed-arity expansion below. Every entry was read against its body;
#: a function missing from here simply keeps its call.
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
        'float': 'py_float', 'int': 'py_int', 'isinstance': 'py_isinstance',
        '_na_of_operands': 'math_na_of_operands',
    },
    'pynecore.lib.array': {
        'NA': 'NA', 'na_float': 'na_float', 'na_int': 'na_int',
        'builtins': 'py_builtins',
        'float': 'py_float', 'int': 'py_int', 'len': 'py_len',
        '_na_element': 'array_na_element',
    },
}

#: Positional arities the varargs expansion is written for. Pine's own
#: ``math.max``/``math.min`` calls are two- or three-operand in practice;
#: anything wider keeps its call.
_VARARGS_ARITIES = (2, 3)


def varargs_expansion_arities() -> tuple[int, ...]:
    """The positional arities :func:`_varargs_template` expands.

    :return: The supported arities.
    """
    return _VARARGS_ARITIES


class NotDerivable(Exception):
    """A wrapper body does not have the shape the pass can copy."""


#: Attribute tagging the ROOT of a subtree that came from the call site's own
#: argument list. Such a subtree is the user's code: it must keep every marker
#: it already had and must never receive the pass's own ones.
_ARG_MARK = '_inl_argument'


def _arg_copy(node: ast.expr) -> ast.expr:
    """A detached copy of an actual argument, tagged as the user's own.

    :param node: The argument expression.
    :return: The tagged copy.
    """
    copied = _deep_copy(node)
    setattr(copied, '_inl_argument', True)
    return copied


def _mark_exact(node: ast.expr) -> ast.expr:
    """Mark the comparisons the pass itself emitted as already-exact.

    The wrapper bodies compare raw -- ``not (x == x)`` is their na test and
    ``exponent == 2`` is a measured shortcut -- so
    ``FloatToleranceTransformer`` must leave those alone. An ARGUMENT subtree
    is the user's own code, though: its comparisons are Pine comparisons and
    must still get the tolerant rewrite, so the walk stops at every tagged
    root. A nested inlined call arrives as such a subtree with its own body
    already marked, which is exactly right.

    :param node: The emitted expression.
    :return: The same expression.
    """
    if getattr(node, _ARG_MARK, False):
        return node
    if isinstance(node, ast.Compare):
        setattr(node, 'pine_exact', True)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _mark_exact(child)
    return node


def _func_path(func: ast.expr) -> str | None:
    """The dotted path of a callee expression, or None if it is not a plain
    name/attribute chain.

    :param func: The callee expression.
    :return: The dotted path.
    """
    if isinstance(func, ast.Name):
        return func.id
    parts: list[str] = []
    current: ast.expr = func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name) and parts:
        parts.append(current.id)
        return '.'.join(reversed(parts))
    return None


# --- source derivation -------------------------------------------------------

#: Parsed wrapper modules, keyed by ``(path, mtime_ns, size)``: a long-lived
#: process that re-imports scripts after a lib edit must re-read the body, not
#: answer from a memo of the previous source.
_module_asts: dict[tuple[str, int, int], dict[str, ast.FunctionDef]] = {}


def _module_functions(module_name: str) -> dict[str, ast.FunctionDef]:
    """Parse a defining module once and index its undecorated top-level defs.

    A decorated definition is skipped on purpose: the runtime name is the
    decorator's return value (an ``overload`` dispatcher), not the body below
    it, so copying that body would be copying the wrong function.

    :param module_name: Importable module name.
    :return: Function name -> its last undecorated definition.
    :raises NotDerivable: If the module has no readable source -- a pyc-only
        install, a zipapp or a frozen module. The site then keeps its call
        instead of failing the whole import.
    """
    try:
        module = importlib.import_module(module_name)
        path = Path(cast(str, module.__file__))
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        cached = _module_asts.get(key)
        if cached is not None:
            return cached
        source = path.read_text(encoding='utf-8')
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise NotDerivable(f"{module_name} has no readable source: {exc!r}") from exc
    functions: dict[str, ast.FunctionDef] = {}
    for stmt in ast.parse(source).body:
        if isinstance(stmt, ast.FunctionDef) and not stmt.decorator_list:
            functions[stmt.name] = stmt
    _module_asts[key] = functions
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
    import typing
    return getattr(module, 'cast', None) is typing.cast


class _BodySubstitutor(ast.NodeTransformer):
    """Rewrite a copied body expression into call-site terms.

    Parameters and body-local aliases become the expressions bound for them,
    every other free name becomes its :mod:`inline_support` anchor, and
    ``typing.cast`` calls fold to their value argument.
    """

    def __init__(self, bindings: dict[str, ast.expr], supported: dict[str, str],
                 module: types.ModuleType, used: set[str]):
        self.bindings = bindings
        self.supported = supported
        self.module = module
        self.used = used

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
        self.used.add(anchor)
        return ast.Name(id=_SUPPORT_ALIAS.format(anchor), ctx=ast.Load())


def _deep_copy(node: ast.expr) -> ast.expr:
    """Structural deep copy of an expression node.

    :param node: The expression.
    :return: A copy sharing no node with the original.
    """
    def walk(value: Any) -> Any:
        if isinstance(value, ast.AST):
            new = type(value).__new__(type(value))
            fields = set(value._fields)
            for field, item in ast.iter_fields(value):
                setattr(new, field, walk(item))
            # Everything the other passes hung on the node travels with it:
            # the Pine type stamp, the exactness marker, the node id. Losing a
            # stamp would silently retype a preserved operand
            for key, item in vars(value).items():
                if key not in fields:
                    setattr(new, key, item)
            return new
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return cast(ast.expr, walk(node))


class _Template:
    """A wrapper body reduced to parameters plus a renderable expression."""

    def __init__(self, params: list[str], render: Any):
        self.params = params
        self._render = render

    def render(self, args: list[ast.expr], used: set[str]) -> ast.expr:
        """Build the call-site expression.

        :param args: One expression per parameter, already bound if needed.
        :param used: Collector of the support anchors the emission needs.
        :return: The inlined expression.
        """
        return cast(ast.expr, self._render(args, used))


def derive_template(module_name: str, func_name: str) -> _Template:
    """Reduce a wrapper's body to a renderable conditional expression.

    Accepted shape, in order: an optional docstring, then any sequence of
    ``if <test>: return <expr>`` guards (no ``else`` branch needed) and
    ``<name> = <expr>`` aliases whose value folds to a name already in scope
    or to a constant, closed by a final ``return <expr>``.

    :param module_name: The defining module.
    :param func_name: The function name.
    :return: The template.
    :raises NotDerivable: If the signature or the body is outside the shape.
    """
    kind = INLINABLE_FUNCTIONS.get((module_name, func_name))
    if kind == 'varargs':
        return _varargs_template(module_name, func_name)
    functions = _module_functions(module_name)
    node = functions.get(func_name)
    if node is None:
        raise NotDerivable(f"{module_name}.{func_name} has no undecorated definition")
    args = node.args
    if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults:
        raise NotDerivable("only plain positional parameters are supported")
    params = [arg.arg for arg in args.args]
    module = importlib.import_module(module_name)
    supported = SUPPORT_NAMES.get(module_name, {})

    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        raise NotDerivable("empty body")

    def render(call_args: list[ast.expr], used: set[str]) -> ast.expr:
        bindings = dict(zip(params, call_args))

        def substitute(expr: ast.expr) -> ast.expr:
            return _BodySubstitutor(bindings, supported, module, used).visit(_deep_copy(expr))

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
                if not isinstance(folded, (ast.Name, ast.Constant)):
                    raise NotDerivable("an alias must fold to a name or a constant")
                target = cast(ast.Name, head.targets[0]).id
                bindings[target] = bindings[folded.id] if isinstance(folded, ast.Name) \
                    else folded
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

        return build(body)

    # Render once with placeholder parameters: a body outside the shape must
    # fail here, at allow-list time, not at the first call site
    render([ast.Name(id=param, ctx=ast.Load()) for param in params], set())
    return _Template(params, render)


def _varargs_shape(module_name: str, func_name: str) -> str:
    """Verify that ``math.max``/``math.min`` still have the body the expansion
    below reproduces, and report which builtin they end in.

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
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
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


def _varargs_template(module_name: str, func_name: str) -> _Template:
    """The fixed-arity expansion of ``math.max`` / ``math.min``.

    For ``max(a, b)`` the body scans the operands for na and otherwise hands
    the whole tuple to ``builtins.max``. Written out, with the operands already
    bound::

        _na_of_operands((a, b)) if not (a == a) or not (b == b)
        else builtins.max((a, b))

    The ``assert numbers`` of the body is dropped: a fixed arity of two or
    three can never fail it. The na scan keeps the loop's left-to-right order
    and short-circuits exactly where the loop's ``return`` does, and the
    builtin is handed the same single tuple argument the body hands it.

    :param module_name: The defining module.
    :param func_name: The function name.
    :return: The template, for the widest supported arity.
    :raises NotDerivable: If the real body drifted from the expected shape.
    """
    builtin_name = _varargs_shape(module_name, func_name)
    supported = SUPPORT_NAMES[module_name]

    def render(call_args: list[ast.expr], used: set[str]) -> ast.expr:
        if len(call_args) not in _VARARGS_ARITIES:
            raise NotDerivable("unsupported arity")
        used.add(supported['_na_of_operands'])
        used.add(supported['builtins'])
        na_anchor = _SUPPORT_ALIAS.format(supported['_na_of_operands'])
        builtins_anchor = _SUPPORT_ALIAS.format(supported['builtins'])

        def operands() -> ast.expr:
            return ast.Tuple(elts=[_arg_copy(arg) for arg in call_args], ctx=ast.Load())

        probes: list[ast.expr] = [
            ast.UnaryOp(op=ast.Not(), operand=ast.Compare(
                left=_arg_copy(arg), ops=[ast.Eq()], comparators=[_arg_copy(arg)]))
            for arg in call_args]
        return ast.IfExp(
            test=ast.BoolOp(op=ast.Or(), values=probes),
            body=ast.Call(func=ast.Name(id=na_anchor, ctx=ast.Load()),
                          args=[operands()], keywords=[]),
            orelse=ast.Call(
                func=ast.Attribute(value=ast.Name(id=builtins_anchor, ctx=ast.Load()),
                                   attr=builtin_name, ctx=ast.Load()),
                args=[operands()], keywords=[]))

    return _Template([f'__arg{i}' for i in range(_VARARGS_ARITIES[-1])], render)


# --- constant folding --------------------------------------------------------

#: Anchors whose one-argument form may be evaluated at transform time over a
#: numeric literal. The fold RUNS the very same callable the emission would
#: have run, so it cannot reason its way to a different int-vs-float result.
_FOLDABLE_UNARY: dict[str, Any] = {'py_int': builtins.int, 'py_float': builtins.float}

#: Comparison operators a fold may evaluate over two numeric literals
_FOLDABLE_COMPARE: dict[type[ast.cmpop], Any] = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _anchor_of(name: str) -> str | None:
    """The support attribute an emitted alias stands for.

    :param name: An emitted name id.
    :return: The anchor name, or None when the id is not an anchor alias.
    """
    if name.startswith(SUPPORT_ALIAS_PREFIX) and name.endswith('__'):
        return name[len(SUPPORT_ALIAS_PREFIX):-2]
    return None


def _literal(node: ast.expr) -> tuple[bool, Any]:
    """The numeric value of a literal operand.

    Only a plain ``int``/``float`` constant counts, optionally negated -- a
    ``bool`` does not (Pine's ``true`` is not the number 1 for the wrappers'
    overloads), a ``str`` does not, and neither does a nan, whose comparisons
    are what the na guards are testing in the first place.

    :param node: The operand.
    :return: ``(True, value)`` for a numeric literal, ``(False, None)`` else.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) is int:
            return True, value
        if type(value) is float and value == value:
            return True, value
        return False, None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ok, value = _literal(node.operand)
        return (True, -value) if ok else (False, None)
    return False, None


class _LiteralFolder(ast.NodeTransformer):
    """Evaluate the parts of a copied body a literal argument already decides.

    ``math.pow(x, 2)`` copies in ``isinstance(2, NA)`` and ``2 != 2``;
    ``array.get(a, 0)`` copies in ``int(0)``. Each is replaced by the result of
    running that exact operation on that exact literal, so the emission cannot
    drift from what the body would have computed. A literal has no side effect,
    so nothing is lost by not evaluating it at runtime.
    """

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if node.keywords or not isinstance(node.func, ast.Name):
            return node
        anchor = _anchor_of(node.func.id)
        if anchor is None:
            return node
        if anchor == 'py_isinstance' and len(node.args) == 2 \
                and isinstance(node.args[1], ast.Name) \
                and _anchor_of(cast(ast.Name, node.args[1]).id) == 'NA':
            ok, value = _literal(node.args[0])
            if ok:
                return ast.Constant(value=builtins.isinstance(value, NA))
            return node
        callable_ = _FOLDABLE_UNARY.get(anchor)
        if callable_ is not None and len(node.args) == 1:
            ok, value = _literal(node.args[0])
            if ok:
                try:
                    return ast.Constant(value=callable_(value))
                except (OverflowError, ValueError):
                    # float() of an int beyond the double range: the fold stays
                    return node
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)
        values: list[Any] = []
        for operand in [node.left, *node.comparators]:
            ok, value = _literal(operand)
            if not ok:
                return node
            values.append(value)
        # A chain is the conjunction of its adjacent pairs
        result = True
        for op, left, right in zip(node.ops, values, values[1:]):
            compare = _FOLDABLE_COMPARE.get(type(op))
            if compare is None:
                return node
            result = result and compare(left, right)
        return ast.Constant(value=result)


def _bool_constant(node: ast.expr) -> bool | None:
    """The value of a boolean constant node.

    :param node: The candidate.
    :return: Its value, or None when it is not a ``bool`` constant.
    """
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return node.value
    return None


def _simplify_test(node: ast.expr) -> ast.expr:
    """Simplify a guard test whose operands the literal fold has decided.

    Only rewrites that cannot change the outcome are applied: a ``not`` over a
    constant, dropping the values a ``BoolOp`` cannot be decided by (``True``
    in an ``and``, ``False`` in an ``or``), and collapsing the whole operator
    when its FIRST remaining value short-circuits -- nothing is evaluated
    before that one, so nothing can be lost.

    :param node: The guard test.
    :return: The simplified test.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = _simplify_test(node.operand)
        decided = _bool_constant(operand)
        if decided is not None:
            return ast.Constant(value=not decided)
        node.operand = operand
        return node
    if isinstance(node, ast.BoolOp):
        identity = isinstance(node.op, ast.And)
        kept: list[ast.expr] = []
        for value in node.values:
            simplified = _simplify_test(value)
            if _bool_constant(simplified) is identity:
                continue
            kept.append(simplified)
        if not kept:
            return ast.Constant(value=identity)
        first = _bool_constant(kept[0])
        if first is not None:
            return kept[0]
        if len(kept) == 1:
            return kept[0]
        node.values = kept
        return node
    return node


def _fold_guards(node: ast.expr) -> ast.expr:
    """Drop the guards of an inlined body that a literal argument decides.

    :param node: The rendered expression.
    :return: The expression with its constant guards resolved.
    """
    while isinstance(node, ast.IfExp):
        node.test = _simplify_test(node.test)
        decided = _bool_constant(node.test)
        if decided is None:
            node.body = _fold_guards(node.body)
            node.orelse = _fold_guards(node.orelse)
            return node
        node = node.body if decided else node.orelse
    return node


# --- argument purity and evaluation order ------------------------------------

#: An argument that is a constant: substitutable anywhere, no side effect, no
#: way to raise.
_ARG_CONST = 'const'
#: A plain name read. No side effect, but it can raise ``NameError`` /
#: ``UnboundLocalError``, so it is ordered against the impure arguments.
_ARG_NAME = 'name'
#: A slot read the Series/Persistent lowering emitted (``__state__[7]``): a
#: plain list index on a local list with a compile-time-allocated index. It
#: cannot run user code and cannot raise, so it may be re-evaluated and needs
#: no ordering.
_ARG_SLOT = 'slot'
#: Anything else. Must be evaluated exactly once, in source order.
_ARG_IMPURE = 'impure'


def _is_state_name(name: str) -> bool:
    """Whether a name is a state vector parameter of the slot lowering.

    :param name: The name id.
    :return: True for ``__state__`` and the per-scope ``__state·<scope>__``.
    """
    return name == DEFAULT_STATE_PARAM or (name.startswith('__state·')
                                           and name.endswith('__'))


def _arg_class(node: ast.expr) -> str:
    """Classify an actual argument by what re-evaluating it would cost.

    :param node: The argument expression.
    :return: One of the ``_ARG_*`` classes.
    """
    if isinstance(node, ast.Constant):
        return _ARG_CONST
    if isinstance(node, ast.Name):
        return _ARG_NAME
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
            and _is_state_name(node.value.id) and isinstance(node.slice, ast.Constant) \
            and type(node.slice.value) is int:
        return _ARG_SLOT
    return _ARG_IMPURE


def _always_nodes(node: ast.expr, out: list[ast.Name]) -> None:
    """Collect the names an expression reads on EVERY path, in order.

    Only the node kinds the emission can produce are walked; anything else
    stops the collection, which only ever makes the caller more conservative.

    :param node: The expression.
    :param out: Accumulator of the name nodes, in evaluation order.
    """
    if isinstance(node, ast.Name):
        out.append(node)
    elif isinstance(node, ast.Constant):
        pass
    elif isinstance(node, ast.NamedExpr):
        _always_nodes(node.value, out)
        if isinstance(node.target, ast.Name):
            out.append(node.target)
    elif isinstance(node, ast.UnaryOp):
        _always_nodes(node.operand, out)
    elif isinstance(node, ast.BinOp):
        _always_nodes(node.left, out)
        _always_nodes(node.right, out)
    elif isinstance(node, ast.BoolOp):
        _always_nodes(node.values[0], out)
    elif isinstance(node, ast.Compare):
        _always_nodes(node.left, out)
        _always_nodes(node.comparators[0], out)
    elif isinstance(node, ast.IfExp):
        _always_nodes(node.test, out)
    elif isinstance(node, ast.Call):
        _always_nodes(node.func, out)
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                return
            _always_nodes(arg, out)
    elif isinstance(node, ast.Attribute):
        _always_nodes(node.value, out)
    elif isinstance(node, ast.Subscript):
        _always_nodes(node.value, out)
        if isinstance(node.slice, ast.expr):
            _always_nodes(node.slice, out)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                return
            _always_nodes(elt, out)


def _replace_node(root: ast.AST, old: ast.AST, new: ast.AST) -> bool:
    """Swap one node for another, found by identity.

    :param root: The tree to search.
    :param old: The node to replace.
    :param new: Its replacement.
    :return: True when the node was found and replaced.
    """
    for field, value in ast.iter_fields(root):
        if value is old:
            setattr(root, field, new)
            return True
        if isinstance(value, list):
            for index, item in enumerate(value):
                if item is old:
                    value[index] = new
                    return True
                if isinstance(item, ast.AST) and _replace_node(item, old, new):
                    return True
        elif isinstance(value, ast.AST) and _replace_node(value, old, new):
            return True
    return False


class _PlaceholderSubstitutor(ast.NodeTransformer):
    """Replace every read of one placeholder with a freshly built expression."""

    def __init__(self, name: str, factory: Any):
        self.name = name
        self.factory = factory

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == self.name:
            return cast(ast.expr, self.factory())
        return node


def _probe(value: ast.expr, name: str) -> ast.expr:
    """``<value> is <name>`` -- an always-true test that pins WHEN the value is
    evaluated.

    :param value: The expression to evaluate (a binding or a plain name read).
    :param name: The name the result is readable under.
    :return: The probe, marked exact.
    """
    probe = ast.Compare(left=value, ops=[ast.Is()],
                        comparators=[ast.Name(id=name, ctx=ast.Load())])
    setattr(probe, 'pine_exact', True)
    return probe


def _chain(probes: list[ast.expr], body: ast.expr) -> ast.expr:
    """Force the argument evaluations to run, in source order, before the body.

    ``x is x`` is True for every object, so the ``and`` always yields the body
    -- including a falsy one, which is what ``and`` returns when its left side
    holds. The form exists because a copied body does not necessarily read the
    parameters in argument order, while the call it replaces evaluated all of
    them, left to right, before entering.

    :param probes: The per-argument probes, in source order.
    :param body: The inlined expression.
    :return: The guarded expression.
    """
    result = body
    for probe in reversed(probes):
        result = ast.BoolOp(op=ast.And(), values=[probe, result])
    return result


# --- the pass ----------------------------------------------------------------


class _NameIndex(ast.NodeVisitor):
    """Every name the module binds anywhere, plus its module-level import map.

    A base name the module binds somewhere is out of the pass: the callee
    could be a user's own ``abs`` or a variable named ``math``, and proving
    otherwise per scope buys nothing here.

    Only an import that is a DIRECT statement of the module body is trusted:
    one inside an ``if``, ``try``, ``with`` or a loop binds its name only on
    some paths, so what it names at runtime is not decidable here. Such an
    import makes its name bound instead, which takes it out of the pass.
    Straight-line re-imports of the same name stay last-write-wins.
    """

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.import_map: dict[str, tuple[str, str | None]] = {}
        self._toplevel: set[int] = set()

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self._toplevel.add(id(stmt))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        args = node.args
        for arg in args.args + args.posonlyargs + args.kwonlyargs:
            self.bound.add(arg.arg)
        if args.vararg:
            self.bound.add(args.vararg.arg)
        if args.kwarg:
            self.bound.add(args.kwarg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(cast(ast.FunctionDef, node))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        trusted = id(node) in self._toplevel
        for alias in node.names:
            bound = alias.asname or alias.name.split('.')[0]
            if not trusted:
                self.bound.add(bound)
                continue
            module = alias.name if alias.asname else alias.name.split('.')[0]
            self.import_map[bound] = (module, None)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        trusted = id(node) in self._toplevel and not node.level and bool(node.module)
        for alias in node.names:
            bound = alias.asname or alias.name
            if not trusted:
                self.bound.add(bound)
                continue
            self.import_map[bound] = (cast(str, node.module), alias.name)


class CallInlineTransformer(ast.NodeTransformer):
    """Replace calls to allow-listed Pine builtins with their own body."""

    def __init__(self) -> None:
        self.index = _NameIndex()
        self._used: set[str] = set()
        self._templates: dict[tuple[str, str], _Template | None] = {}
        self._resolved: dict[str, Any] = {}
        self._targets: dict[str, tuple[str, str] | None] = {}
        self._counter = 0
        self._func_depth = 0
        self._blocked = 0
        #: Number of sites rewritten -- the pass's own tests assert on it
        self.inlined = 0

    # --- callee proof ------------------------------------------------------

    def _target(self, func: ast.expr) -> tuple[str, str] | None:
        """Prove that a callee IS an allow-listed library function.

        The answer is memoized per dotted path: an unprovable path would
        otherwise re-walk the whole allow-list on every site.

        :param func: The callee expression.
        :return: The ``(module, name)`` entry, or None when unprovable.
        """
        path = _func_path(func)
        if path is None:
            return None
        try:
            return self._targets[path]
        except KeyError:
            pass
        target = self._resolve_target(path)
        self._targets[path] = target
        return target

    def _resolve_target(self, path: str) -> tuple[str, str] | None:
        """Resolve one dotted path against the allow-list, by object identity.

        :param path: The dotted callee path.
        :return: The ``(module, name)`` entry, or None when unprovable.
        """
        parts = path.split('.')
        base = parts[0]
        if base in self.index.bound:
            return None
        entry = self.index.import_map.get(base)
        if entry is None:
            return None
        obj = self._resolve(path, parts, entry)
        if obj is None:
            return None
        for key in INLINABLE_FUNCTIONS:
            try:
                module = importlib.import_module(key[0])
            except ImportError:
                continue
            if getattr(module, key[1], None) is obj:
                return key
        return None

    def _resolve(self, path: str, parts: list[str],
                 entry: tuple[str, str | None]) -> Any | None:
        """Resolve a dotted callee path through the module-level import map.

        :param path: The dotted path, used as the cache key.
        :param parts: Its segments.
        :param entry: The import map entry of the base name.
        :return: The runtime object, or None if it cannot be resolved.
        """
        try:
            return self._resolved[path]
        except KeyError:
            pass
        module_name, attr = entry
        obj: Any | None
        if module_name != _PACKAGE and not module_name.startswith(_PACKAGE + '.'):
            # Every inlinable function lives in this package, and a user module
            # is not imported just to look at it
            self._resolved[path] = None
            return None
        try:
            obj = importlib.import_module(module_name)
            for name in ([attr] if attr else []) + parts[1:]:
                try:
                    obj = getattr(obj, name)
                except AttributeError:
                    if not isinstance(obj, types.ModuleType):
                        raise
                    obj = importlib.import_module(f'{obj.__name__}.{name}')
        except (ImportError, AttributeError):
            obj = None
        self._resolved[path] = obj
        return obj

    def _template(self, key: tuple[str, str]) -> _Template | None:
        """The template of an allow-listed function, derived once.

        :param key: ``(module, name)``.
        :return: The template, or None when the body is not derivable.
        """
        try:
            return self._templates[key]
        except KeyError:
            pass
        try:
            template: _Template | None = derive_template(*key)
        except NotDerivable:
            template = None
        self._templates[key] = template
        return template

    # --- emission ----------------------------------------------------------

    def _inline(self, node: ast.Call) -> ast.expr | None:
        """Build the inlined expression for a call site, or None to keep it.

        :param node: The call site.
        :return: The replacement expression.
        """
        if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
            return None
        key = self._target(node.func)
        if key is None:
            return None
        template = self._template(key)
        if template is None:
            return None
        if INLINABLE_FUNCTIONS[key] == 'derive':
            if len(node.args) != len(template.params):
                return None
        elif len(node.args) not in _VARARGS_ARITIES:
            return None

        classes = [_arg_class(arg) for arg in node.args]
        # A constant goes in verbatim, so the literal fold can decide the
        # guards it controls; everything else renders as a placeholder and is
        # put in once the emission's shape -- and with it the read count and
        # the evaluation order -- is known
        placeholders: dict[int, str] = {}
        render_args: list[ast.expr] = []
        for index, arg in enumerate(node.args):
            if classes[index] == _ARG_CONST:
                render_args.append(_arg_copy(arg))
            else:
                name = _PLACEHOLDER.format(index)
                placeholders[index] = name
                render_args.append(ast.Name(id=name, ctx=ast.Load()))

        used: set[str] = set()
        try:
            expression = _fold_guards(_LiteralFolder().visit(template.render(render_args,
                                                                            used)))
        except NotDerivable:
            return None
        # Marked BEFORE the arguments go in: what is in the tree now is the
        # pass's own emission, and the user's subtrees must stay untouched
        _mark_exact(expression)

        reads: dict[int, int] = {index: 0 for index in placeholders}
        for child in ast.walk(expression):
            if isinstance(child, ast.Name):
                for index, name in placeholders.items():
                    if child.id == name:
                        reads[index] += 1
        always: list[ast.Name] = []
        _always_nodes(expression, always)
        first_always: dict[int, ast.Name] = {}
        for name_node in always:
            for index, name in placeholders.items():
                if name_node.id == name and index not in first_always:
                    first_always[index] = name_node

        # A name read can raise, so it is ordered against the impure
        # arguments; a constant and a slot read can do neither
        ordered = [index for index, kind in enumerate(classes)
                   if kind in (_ARG_NAME, _ARG_IMPURE)]
        impure = [index for index, kind in enumerate(classes) if kind == _ARG_IMPURE]
        natural = [index for index in
                   sorted(first_always, key=lambda i: always.index(first_always[i]))
                   if index in ordered]
        force = bool(impure) and natural != ordered

        probes: list[ast.expr] = []
        for index, kind in enumerate(classes):
            if kind == _ARG_CONST:
                continue
            name = placeholders[index]
            argument = node.args[index]
            anchor = first_always.get(index)
            if force and index in ordered:
                if kind == _ARG_NAME:
                    # A name needs no temporary: reading it twice is free and
                    # the probe is only there to pin WHERE it is read
                    probes.append(_probe(_arg_copy(argument),
                                         cast(ast.Name, argument).id))
                    _PlaceholderSubstitutor(
                        name, lambda a=argument: _arg_copy(a)).visit(expression)
                    continue
                self._counter += 1
                temp = _TEMP_NAME.format(self._counter)
                probes.append(_probe(ast.NamedExpr(
                    target=ast.Name(id=temp, ctx=ast.Store()),
                    value=_arg_copy(argument)), temp))
                _PlaceholderSubstitutor(
                    name, lambda t=temp: ast.Name(id=t, ctx=ast.Load())).visit(expression)
                continue
            #  - a name is free to re-read, so it always goes in as itself;
            #  - a slot read is free of side effects but costs an index, so it
            #    is bound only when the body reads it more than once AND the
            #    first read happens on every path;
            #  - anything else must run exactly once, so a single read takes
            #    the expression itself and several reads take a binding.
            direct = kind == _ARG_NAME \
                or (kind == _ARG_SLOT and (reads[index] <= 1 or anchor is None)) \
                or (kind == _ARG_IMPURE and reads[index] <= 1 and anchor is not None)
            if direct:
                _PlaceholderSubstitutor(
                    name, lambda a=argument: _arg_copy(a)).visit(expression)
                continue
            # Read more than once: bound once at its first read, which the
            # order check above proved is on every path
            self._counter += 1
            temp = _TEMP_NAME.format(self._counter)
            _replace_node(expression, cast(ast.Name, anchor), ast.NamedExpr(
                target=ast.Name(id=temp, ctx=ast.Store()), value=_arg_copy(argument)))
            _PlaceholderSubstitutor(
                name, lambda t=temp: ast.Name(id=t, ctx=ast.Load())).visit(expression)

        if probes:
            expression = _chain(probes, expression)

        # Collected from the FINAL expression: a guard folded away takes its
        # anchors with it, and an import nothing reads is dead weight
        del used
        for child in ast.walk(expression):
            if isinstance(child, ast.Name) and child.id.startswith(SUPPORT_ALIAS_PREFIX):
                anchor_name = _anchor_of(child.id)
                if anchor_name is not None:
                    self._used.add(anchor_name)
        self.inlined += 1
        return ast.copy_location(
            stamp_lowering(expression, get_ty(node)), node)

    # --- visitors ----------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.index = _NameIndex()
        self.index.visit(node)
        node = cast(ast.Module, self.generic_visit(node))
        if self._used:
            import_stmt = ast.ImportFrom(
                module=SUPPORT_MODULE,
                names=[ast.alias(name=name, asname=_SUPPORT_ALIAS.format(name))
                       for name in sorted(self._used)],
                level=0)
            insert_pos = 0
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_pos = 1
            for i in range(insert_pos, len(node.body)):
                if isinstance(node.body[i], (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif not isinstance(node.body[i], ast.Expr):
                    break
            node.body.insert(insert_pos, import_stmt)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # A temporary bound in a class body would become a class attribute
        self._blocked += 1
        self.generic_visit(node)
        self._blocked -= 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Decorators and defaults are evaluated in the ENCLOSING scope, so they
        # keep that scope's depth; only the body gets the function's own
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(default)
        self._func_depth += 1
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]
        self._func_depth -= 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.visit_FunctionDef(cast(ast.FunctionDef, node))
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # A walrus in a lambda binds in the lambda's own scope
        self._blocked += 1
        self.generic_visit(node)
        self._blocked -= 1
        return node

    def _visit_comprehension(self, node: ast.expr) -> ast.expr:
        # A walrus is illegal in a comprehension's iterable and binds to the
        # containing scope elsewhere in it
        self._blocked += 1
        self.generic_visit(node)
        self._blocked -= 1
        return node

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if self._blocked or not self._func_depth:
            return node
        return self._inline(node) or node
