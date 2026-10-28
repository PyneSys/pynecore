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
hand-written: ``scripts/call_inline_collector.py`` parses the defining module,
accepts the function body only in a restricted shape (docstring, ``if <test>:
return <expr>`` guards, name aliases, one final ``return <expr>``) and folds it
into a nested conditional expression. The same operations run in the same
order on the same operands, so the result is the same double, the same ``na``
object and the same exception. The derived expressions are the generated
registry ``call_inline_templates.json``, which this pass reads instead of the
lib: the transform imports no lib module and parses no lib source, and a lib
edit that changes a template fails ``test_142`` until the file is regenerated
(see the collector's docstring).

A guard a LITERAL argument decides is resolved at transform time
(``math.pow(x, 2)`` keeps only ``x * x``), and so are the operations that
guard reads: ``2 == 2``, ``int(0)``. Each is folded by RUNNING the very same
operation on that very literal, never by reasoning about it, so an
int-vs-float result cannot drift. A ``bool`` is not treated as
a numeric literal, so nothing folds on ``math.pow(x, true)``.

One fold is not a literal's: the counter of a ``for`` loop over ``pine_range``
is never na, so a self-comparison of it (``i == i``, the na guard of
``array.get(a, i)``) is True. The fact is proven per loop, not assumed -- see
:meth:`CallInlineTransformer._loop_counter`.

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
a structural check of their real bodies in the collector, so an edit to
either body fails the generation instead of changing what the expansion means.

Name resolution
---------------

The copied body keeps its own module's names, which mean something else at
the call site: a script's ``math`` is ``pynecore.lib.math``, and
``BuiltinShadowTransformer`` lets it rebind ``abs`` or ``float``. Every free
name of a copied body therefore resolves through
:mod:`~pynecore.core.inline_support`, imported once per module under the same
collision-safe middle-dot alias the other passes use. A name that module does
not anchor makes the body non-derivable.

The callee itself must be PROVABLY the library function: the base name of the
dotted path is resolved through the module's import map at transform time,
and the resulting path must be one the registry records for the function (its
own module path, or a re-export by the packages scripts import from). A name
the module binds anywhere -- a user ``def abs``, a variable named ``math`` --
takes the whole base name out of the pass.

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
from collections.abc import Callable
from typing import Any, cast
import ast
import builtins
import json
import operator
from pathlib import Path

from . import ast_walk
from .pine_type_rules import get_ty, stamp_lowering
from .state_param import DEFAULT_STATE_PARAM

__all__ = ['CallInlineTransformer', 'SUPPORT_MODULE', 'SUPPORT_ALIAS_PREFIX', 'inline_templates',
           'varargs_expansion_arities']

#: The module every free name of an inlined body resolves through
SUPPORT_MODULE = 'pynecore.core.inline_support'

#: Alias prefix/suffix of an anchored support name, and of a bound argument.
#: The middle dot cannot appear in a Python identifier, so neither a script
#: variable nor another pass's temporary can collide with these.
SUPPORT_ALIAS_PREFIX = '__inl·'
_SUPPORT_ALIAS = SUPPORT_ALIAS_PREFIX + '{}__'
_TEMP_NAME = '__inl{}·__'
#: Stand-in for an actual argument while the emission's shape is decided
_PLACEHOLDER = '__inlp{}·__'

#: The generated template registry (``scripts/call_inline_collector.py``)
_TEMPLATES_PATH = Path(__file__).parent / 'call_inline_templates.json'
_REGISTRY: dict[str, Any] = {}


def inline_templates() -> dict[str, Any]:
    """The generated template registry, read once per process.

    :return: Its sections: ``functions`` (function path -> its template),
        ``callees`` (callee path -> function path), ``loop_ranges`` (the paths
        of ``pine_range``) and ``varargs_arities``.
    """
    if not _REGISTRY:
        _REGISTRY.update(json.loads(_TEMPLATES_PATH.read_text(encoding='utf-8')))
    return _REGISTRY


def varargs_expansion_arities() -> tuple[int, ...]:
    """The positional arities the ``math.max`` / ``math.min`` expansion is written for.

    :return: The supported arities.
    """
    return tuple(inline_templates()['varargs_arities'])


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
    for child in ast_walk.iter_child_nodes(node):
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


def _stamp_emission(node: ast.AST, call: ast.Call) -> None:
    """Anchor every node of the pass's own emission to the call site's point.

    The template is copied from the wrapper's source, so its nodes carry line
    numbers of ``lib/math.py`` or ``lib/array.py`` -- lines the script does
    not have, or has with other code on them. Line events, tracebacks and
    profiles must see the call's own line instead. The argument subtrees are
    the user's code and keep their locations (a nested inlined call was
    anchored when it was inlined). A call another pass built without a
    location leaves the emission unlocated, for ``fix_locations`` to anchor
    with the enclosing statement.

    :param node: The emission root.
    :param call: The call site being replaced.
    """
    line: int | None = getattr(call, 'lineno', None)
    col: int = getattr(call, 'col_offset', 0)
    stack = [node]
    while stack:
        current = stack.pop()
        if getattr(current, _ARG_MARK, False):
            continue
        if 'lineno' in current._attributes:
            if line is None:
                for attribute in current._attributes:
                    current.__dict__.pop(attribute, None)
            else:
                current.lineno = current.end_lineno = line  # type: ignore[attr-defined]
                current.col_offset = current.end_col_offset = col  # type: ignore[attr-defined]
        stack.extend(ast_walk.iter_child_nodes(current))


def _deep_copy(node: ast.expr) -> ast.expr:
    """Structural deep copy of an expression node.

    :param node: The expression.
    :return: A copy sharing no node with the original.
    """
    return cast(ast.expr, _copy_node(node))


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
            item = [_copy_value(entry) for entry in item]
        state[field] = item
    # Everything the other passes hung on the node travels with it: the Pine
    # type stamp, the exactness marker, the node id. Losing a stamp would
    # silently retype a preserved operand
    for key, item in value.__dict__.items():
        if key not in fields:
            state[key] = item
    return new


def _copy_value(value: Any) -> Any:
    """Copy a field value: a node or a list deeply, anything else as it is."""
    if isinstance(value, ast.AST):
        return _copy_node(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


# --- templates ---------------------------------------------------------------

def _constant(spec: list[Any]) -> Any:
    """The value of a template constant (``[type, value]``, see the collector).

    :param spec: The tagged value.
    :return: The value.
    """
    tag, value = spec
    if tag == 'int':
        return int(value)
    if tag == 'float':
        return float(value)
    return value


def _build(spec: dict[str, Any], args: list[ast.expr]) -> ast.AST:
    """Build one node of a derived template for a call site.

    A read of the i-th parameter becomes a tagged copy of the i-th argument;
    every other node is built afresh and unlocated, for :func:`_stamp_emission`
    to anchor at the call site.

    :param spec: The node's encoding.
    :param args: One expression per parameter.
    :return: The node.
    """
    index = spec.get('param')
    if index is not None:
        return _arg_copy(args[index])
    cls = getattr(ast, spec['_type'])
    node = cls.__new__(cls)
    state = node.__dict__
    for field in cls._fields:
        if field not in spec:
            continue
        value = spec[field]
        if field == 'value' and cls is ast.Constant:
            value = _constant(value)
        elif field in ('op', 'ctx'):
            value = getattr(ast, value)()
        elif field == 'ops':
            value = [getattr(ast, op)() for op in value]
        elif isinstance(value, dict):
            value = _build(value, args)
        elif isinstance(value, list):
            value = [_build(item, args) if isinstance(item, dict) else item for item in value]
        state[field] = value
    if spec.get('arg'):
        state[_ARG_MARK] = True
    return node


class _Template:
    """A wrapper body reduced to parameters plus a renderable expression."""

    def __init__(self, kind: str, params: list[str],
                 render: Callable[[list[ast.expr]], ast.expr]):
        #: ``derive`` (the body's own expression) or ``varargs``
        self.kind = kind
        self.params = params
        self._render = render

    def render(self, args: list[ast.expr]) -> ast.expr:
        """Build the call-site expression.

        :param args: One expression per parameter, already bound if needed.
        :return: The inlined expression.
        """
        return self._render(args)


def _derived_template(entry: dict[str, Any]) -> _Template:
    """The template of a ``derive`` registry entry.

    :param entry: The entry.
    :return: The template.
    """
    body = entry['body']
    return _Template('derive', entry['params'],
                     lambda call_args: cast(ast.expr, _build(body, call_args)))


def _varargs_template(entry: dict[str, Any]) -> _Template:
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

    :param entry: The registry entry: the builtin the body ends in and the
        anchors of ``_na_of_operands`` and ``builtins``.
    :return: The template, for the widest supported arity.
    """
    builtin_name = entry['builtin']
    na_anchor = _SUPPORT_ALIAS.format(entry['na_anchor'])
    builtins_anchor = _SUPPORT_ALIAS.format(entry['builtins_anchor'])

    def render(call_args: list[ast.expr]) -> ast.expr:
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

    arities = varargs_expansion_arities()
    return _Template('varargs', [f'__arg{i}' for i in range(arities[-1])], render)


#: Templates built from the registry, process-wide: function path -> template
_TEMPLATES: dict[str, _Template] = {}


def _template(key: str) -> _Template:
    """The template of a registry function, built once per process.

    :param key: The function path (``pynecore.lib.math.abs``).
    :return: The template.
    """
    template = _TEMPLATES.get(key)
    if template is None:
        entry = inline_templates()['functions'][key]
        template = _TEMPLATES[key] = (_derived_template(entry) if entry['kind'] == 'derive'
                                      else _varargs_template(entry))
    return template


# --- constant folding --------------------------------------------------------

#: Anchors whose one-argument form may be evaluated at transform time over a
#: numeric literal. The fold RUNS the very same callable the emission would
#: have run, so it cannot reason its way to a different int-vs-float result.
_FOLDABLE_UNARY: dict[str, Callable[[Any], Any]] = {
    'py_int': builtins.int, 'py_float': builtins.float,
}

#: Comparison operators a fold may evaluate over two numeric literals
_FOLDABLE_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
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


class _LiteralFolder(ast_walk.NodeTransformer):
    """Evaluate the parts of a copied body a literal argument already decides.

    ``math.pow(x, 2)`` copies in ``2 == 2``; ``array.get(a, 0)`` copies in
    ``int(0)``. Each is replaced by the result of running that exact operation
    on that exact literal, so the emission cannot drift from what the body
    would have computed. A literal has no side effect, so nothing is lost by
    not evaluating it at runtime.

    The one fold that is not a literal's: a self-comparison of a ``for`` loop
    counter (``i == i``, the wrappers' na test) -- see
    :meth:`CallInlineTransformer._loop_counter` for why it is always true.
    """

    def __init__(self, counters: frozenset[str] = frozenset()):
        #: Placeholder names standing for a proven loop counter argument
        self.counters = counters

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if node.keywords or not isinstance(node.func, ast.Name):
            return node
        anchor = _anchor_of(node.func.id)
        if anchor is None:
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
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)) \
                and isinstance(node.left, ast.Name) and node.left.id in self.counters \
                and isinstance(node.comparators[0], ast.Name) \
                and node.comparators[0].id == node.left.id:
            return ast.Constant(value=isinstance(node.ops[0], ast.Eq))
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


class _PlaceholderSubstitutor(ast_walk.NodeTransformer):
    """Replace every read of a placeholder with a freshly built expression."""

    def __init__(self, factories: dict[str, Callable[[], ast.expr]]):
        #: Placeholder name -> the factory building its replacement
        self.factories = factories

    def visit_Name(self, node: ast.Name) -> ast.expr:
        factory = self.factories.get(node.id)
        if factory is not None:
            return factory()
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


class _NameIndex(ast_walk.NodeVisitor):
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


class _Rebinds(ast_walk.NodeVisitor):
    """Find whether a subtree binds one name: a store or delete, a definition,
    an ``except ... as``, an import alias or a match capture."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and not isinstance(node.ctx, ast.Load):
            self.found = True

    def _visit_named(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                     | ast.ExceptHandler | ast.MatchAs | ast.MatchStar) -> None:
        if node.name == self.name:
            self.found = True
        self.generic_visit(node)

    visit_FunctionDef = _visit_named
    visit_AsyncFunctionDef = _visit_named
    visit_ClassDef = _visit_named
    visit_ExceptHandler = _visit_named
    visit_MatchAs = _visit_named
    visit_MatchStar = _visit_named

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        if any((alias.asname or alias.name.split('.')[0]) == self.name for alias in node.names):
            self.found = True

    visit_ImportFrom = visit_Import


class CallInlineTransformer(ast_walk.NodeTransformer):
    """Replace calls to allow-listed Pine builtins with their own body."""

    def __init__(self) -> None:
        self.index = _NameIndex()
        self._used: set[str] = set()
        self._counter = 0
        self._func_depth = 0
        #: The enclosing function definitions, innermost last
        self._functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        #: The loop counters proven never na at the current position
        self._counters: frozenset[str] = frozenset()
        #: id(function definition) -> the names it declares ``global`` or
        #: ``nonlocal`` anywhere inside it (the pass adds no such declaration,
        #: so one scan per function answers for every loop in it)
        self._declared: dict[int, frozenset[str]] = {}
        #: Number of sites rewritten -- the pass's own tests assert on it
        self.inlined = 0

    # --- callee proof ------------------------------------------------------

    def _resolve_path(self, func: ast.expr) -> str | None:
        """The path a callee spells once the module's import map resolved its base name.

        :param func: The callee expression.
        :return: The resolved dotted path, or None when the base name is bound by
            the module or imported by no trusted import.
        """
        path = _func_path(func)
        if path is None:
            return None
        parts = path.split('.')
        base = parts[0]
        if base in self.index.bound:
            return None
        entry = self.index.import_map.get(base)
        if entry is None:
            return None
        module_name, attr = entry
        return '.'.join([module_name, *([attr] if attr else []), *parts[1:]])

    def _target(self, func: ast.expr) -> str | None:
        """Prove that a callee IS an allow-listed library function.

        :param func: The callee expression.
        :return: The function's registry path, or None when unprovable.
        """
        path = self._resolve_path(func)
        if path is None:
            return None
        return inline_templates()['callees'].get(path)

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
        template = _template(key)
        if template.kind == 'derive':
            if len(node.args) != len(template.params):
                return None
        elif len(node.args) not in varargs_expansion_arities():
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

        counters = frozenset(placeholders[index] for index, kind in enumerate(classes)
                             if kind == _ARG_NAME
                             and cast(ast.Name, node.args[index]).id in self._counters)
        expression = _fold_guards(_LiteralFolder(counters).visit(template.render(render_args)))
        # Marked BEFORE the arguments go in: what is in the tree now is the
        # pass's own emission, and the user's subtrees must stay untouched
        _mark_exact(expression)

        # The anchors are collected here, from the FOLDED body: a guard folded
        # away takes its anchors with it, and an import nothing reads is dead
        # weight. The arguments going in below bring none of their own -- a
        # nested inlined call registered its anchors when it was inlined
        reads: dict[int, int] = {index: 0 for index in placeholders}
        indexes = {name: index for index, name in placeholders.items()}
        for child in ast_walk.walk(expression):
            if isinstance(child, ast.Name):
                index = indexes.get(child.id)
                if index is not None:
                    reads[index] += 1
                elif child.id.startswith(SUPPORT_ALIAS_PREFIX):
                    anchor_name = _anchor_of(child.id)
                    if anchor_name is not None:
                        self._used.add(anchor_name)
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

        # Every placeholder is put in by ONE substitution pass at the end: the
        # replacements hold no placeholder, so the order they go in is moot
        factories: dict[str, Callable[[], ast.expr]] = {}
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
                    factories[name] = lambda a=argument: _arg_copy(a)
                    continue
                self._counter += 1
                temp = _TEMP_NAME.format(self._counter)
                probes.append(_probe(ast.NamedExpr(
                    target=ast.Name(id=temp, ctx=ast.Store()),
                    value=_arg_copy(argument)), temp))
                factories[name] = lambda t=temp: ast.Name(id=t, ctx=ast.Load())
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
                factories[name] = lambda a=argument: _arg_copy(a)
                continue
            # Read more than once: bound once at its first read, which the
            # order check above proved is on every path
            self._counter += 1
            temp = _TEMP_NAME.format(self._counter)
            _replace_node(expression, cast(ast.Name, anchor), ast.NamedExpr(
                target=ast.Name(id=temp, ctx=ast.Store()), value=_arg_copy(argument)))
            factories[name] = lambda t=temp: ast.Name(id=t, ctx=ast.Load())
        if factories:
            _PlaceholderSubstitutor(factories).visit(expression)

        if probes:
            expression = _chain(probes, expression)

        self.inlined += 1
        _stamp_emission(expression, node)
        return ast.copy_location(
            stamp_lowering(expression, get_ty(node)), node)

    # --- visitors ----------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.index = _NameIndex()
        self.index.visit(node)
        self._visit_definitions(node)
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

    def _visit_definitions(self, node: ast.AST) -> None:
        """Visit the function definitions among the statements under ``node``.

        Outside a function body nothing is inlined, so no expression there is
        entered; a definition is a statement, and the walk only descends
        through statements to reach every one of them. A class body is not
        entered (see ``visit_ClassDef``).

        :param node: The module, or a statement outside every function.
        """
        for child in ast_walk.iter_child_statements(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
            elif not isinstance(child, ast.ClassDef):
                self._visit_definitions(child)

    def _loop_counter(self, node: ast.For) -> str | None:
        """The counter of a loop whose value is never na anywhere in its body.

        ``pine_range`` only ever yields a value that passed its ``<= to`` /
        ``>= to`` test (a nan or NA bound fails it and ends the loop), so the
        counter starts every iteration as a real number. It stays one through
        the body when nothing can rebind it there: no store to the name
        anywhere in the body (nested scopes included, to stay simple), and no
        ``global`` / ``nonlocal`` declaration of it in the enclosing function,
        which is what would let a called closure or another function rebind
        it mid-body.

        :param node: The loop.
        :return: The counter name, or None when the fact is not proven.
        """
        if not self._functions or not isinstance(node.target, ast.Name) \
                or not isinstance(node.iter, ast.Call):
            return None
        if self._resolve_path(node.iter.func) not in inline_templates()['loop_ranges']:
            return None
        name = node.target.id
        rebinds = _Rebinds(name)
        for stmt in node.body:
            rebinds.visit(stmt)
        if rebinds.found or name in self._declared_names(self._functions[-1]):
            return None
        return name

    def _declared_names(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
        """The names a function declares ``global`` / ``nonlocal``, nested scopes included.

        :param func: The function definition.
        :return: Every name of every such declaration inside it.
        """
        declared = self._declared.get(id(func))
        if declared is None:
            declared = frozenset(name for child in ast_walk.walk_statements(func)
                                 if isinstance(child, (ast.Global, ast.Nonlocal))
                                 for name in child.names)
            self._declared[id(func)] = declared
        return declared

    def visit_For(self, node: ast.For) -> ast.For:
        node.target = cast(ast.expr, self.visit(node.target))
        node.iter = cast(ast.expr, self.visit(node.iter))
        outer = self._counters
        counter = self._loop_counter(node)
        if counter is not None:
            self._counters = outer | {counter}
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]
        self._counters = outer
        node.orelse = [cast(ast.stmt, self.visit(stmt)) for stmt in node.orelse]
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # A temporary bound in a class body would become a class attribute, so
        # no site in it is inlined -- a method body included -- and the subtree
        # is not entered at all
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Decorators and defaults are evaluated in the ENCLOSING scope, so they
        # keep that scope's depth; only the body gets the function's own
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(default)
        # A nested function may run after the loop that encloses it rebound
        # the counter, so no outer counter fact carries into its body
        outer = self._counters
        self._counters = frozenset()
        self._functions.append(node)
        self._func_depth += 1
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]
        self._func_depth -= 1
        self._functions.pop()
        self._counters = outer
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.visit_FunctionDef(cast(ast.FunctionDef, node))
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        # A walrus in a lambda binds in the lambda's own scope, so no site in
        # it is inlined and it is not entered
        return node

    def _visit_comprehension(self, node: ast.expr) -> ast.expr:
        # A walrus is illegal in a comprehension's iterable and binds to the
        # containing scope elsewhere in it, so no site in it is inlined and it
        # is not entered
        return node

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if not self._func_depth:
            return node
        return self._inline(node) or node
