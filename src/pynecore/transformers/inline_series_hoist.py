"""
Hoist ``inline_series`` calls out of lazily evaluated expression positions.

Pine evaluates a history-referenced expression (``expr[n]``) on every bar its
statement executes, even when the ``[n]`` sits in a ternary branch or in a
short-circuited ``and``/``or`` operand the bar's values skip. Verified against
TradingView v6 bar-by-bar: ternary branches and ``and``/``or`` operands always
yield fresh history; only statements inside conditionally executed BLOCKS
(``if`` bodies) keep the documented compressed "gap" history. PyneComp
compiles such history references to ``inline_series(expr, n)``, whose
per-anchor buffer advances only when the call site is actually reached — so
inside a lazy context it returns STALE history after skipped bars.

This pass hoists every ``inline_series(...)`` call found in a lazy expression
position to a temp assignment placed immediately before the enclosing
statement and replaces the call with the temp's name. The assignment stays in
the same statement list, so block-level conditional execution keeps its gap
semantics and loop bodies keep their per-iteration frequency.

``while`` tests are deliberately left untouched (they re-evaluate per
iteration — a hoist above the loop would freeze them), and lambdas /
comprehensions are not descended into (deferred or repeated evaluation).
"""
import ast
from typing import cast

from pynecore.transformers.locations import fix_locations

__all__ = ['InlineSeriesHoistTransformer']


def _is_inline_series(node: ast.expr) -> bool:
    """Whether a node is a direct ``inline_series(...)`` call.

    :param node: Expression node to check.
    :return: True for a plain-name ``inline_series`` call (PyneComp's emission).
    """
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'inline_series')


class _ExprHoister(ast.NodeTransformer):
    """Rewrite one statement's expression tree: every ``inline_series`` call
    in a lazy position is appended to ``hoisted`` as a temp assignment and
    replaced by the temp's name.

    ``_lazy`` is True while the current subtree only evaluates when
    short-circuiting (``and``/``or`` operands after the first) or ternary
    branch selection reaches it. Inner calls are collected before outer ones,
    so the emitted assignments are ordered definition-before-use.
    """

    def __init__(self, transformer: 'InlineSeriesHoistTransformer'):
        self._transformer = transformer
        self.hoisted: list[ast.stmt] = []
        self._lazy = False

    def visit_as(self, node: ast.expr, lazy: bool) -> ast.expr:
        """Visit a subtree with an explicit laziness flag.

        :param node: Subtree root.
        :param lazy: Whether the subtree is in a lazily evaluated position.
        :return: The (possibly rewritten) subtree.
        """
        prev, self._lazy = self._lazy, lazy
        try:
            return cast(ast.expr, self.visit(node))
        finally:
            self._lazy = prev

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        node.values = [self.visit_as(value, self._lazy if i == 0 else True)
                       for i, value in enumerate(node.values)]
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        node.test = self.visit_as(node.test, self._lazy)
        node.body = self.visit_as(node.body, True)
        node.orelse = self.visit_as(node.orelse, True)
        return node

    # Deferred / repeated evaluation contexts: hoisting from inside would
    # change how often the call runs — leave them untouched.
    def visit_Lambda(self, node: ast.Lambda) -> ast.expr:
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.expr:
        return node

    def visit_SetComp(self, node: ast.SetComp) -> ast.expr:
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.expr:
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.expr:
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if not (self._lazy and _is_inline_series(node)):
            return cast(ast.expr, self.generic_visit(node))
        # The hoisted call runs at statement level: its arguments leave the
        # lazy context, so nested positions are judged from an eager root
        node.args = [self.visit_as(arg, False) for arg in node.args]
        for keyword in node.keywords:
            keyword.value = self.visit_as(keyword.value, False)
        name = self._transformer.next_name()
        self.hoisted.append(ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=node,
        ))
        return ast.Name(id=name, ctx=ast.Load())


class InlineSeriesHoistTransformer:
    """Statement-list walker applying :class:`_ExprHoister` to every
    statement's own expressions, prepending the hoisted temp assignments in
    the same statement list."""

    #: Fields holding nested statement lists (handlers are special-cased).
    _STMT_LIST_FIELDS = ('body', 'orelse', 'finalbody')

    #: Statements whose own expressions evaluate at definition time
    #: (decorators, defaults, bases) — nothing to hoist per bar.
    _SKIP_EXPR_STMTS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def __init__(self):
        self._counter = 0

    def next_name(self) -> str:
        """Allocate a module-unique temp name for a hoisted call."""
        name = f'__hist_{self._counter}__'
        self._counter += 1
        return name

    def visit(self, tree: ast.Module) -> ast.Module:
        """Process a module tree in place.

        :param tree: Parsed module.
        :return: The same tree with lazy ``inline_series`` calls hoisted.
        """
        tree.body = self._process_body(tree.body)
        if self._counter:
            fix_locations(tree)
        return tree

    def _process_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Process one statement list; returns the list with hoists inserted."""
        new_body: list[ast.stmt] = []
        for stmt in body:
            # Nested statement lists first — each is its own hoist target,
            # which keeps block-level (gap) semantics intact
            for field in self._STMT_LIST_FIELDS:
                value = getattr(stmt, field, None)
                if value and isinstance(value[0], ast.stmt):
                    setattr(stmt, field, self._process_body(value))
            for handler in getattr(stmt, 'handlers', ()):
                handler.body = self._process_body(handler.body)
            new_body.extend(self._hoist_from(stmt))
            new_body.append(stmt)
        return new_body

    def _hoist_from(self, stmt: ast.stmt) -> list[ast.stmt]:
        """Rewrite the statement's own expression fields; return the temp
        assignments to place before it."""
        if isinstance(stmt, self._SKIP_EXPR_STMTS):
            return []
        hoister = _ExprHoister(self)
        for field, value in ast.iter_fields(stmt):
            if isinstance(stmt, ast.While) and field == 'test':
                continue  # re-evaluated per iteration — must stay in place
            if isinstance(value, ast.expr):
                setattr(stmt, field, hoister.visit_as(value, False))
            elif (isinstance(value, list) and value
                  and isinstance(value[0], ast.expr)):
                setattr(stmt, field,
                        [hoister.visit_as(item, False) for item in value])
        return hoister.hoisted
