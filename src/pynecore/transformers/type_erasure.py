"""Erase runtime calls that exist only for the static type checker.

``typing.cast(T, x)`` returns ``x`` unchanged, yet it is a real Python call: a
frame-less C builtin would be cheap, ``typing.cast`` is a Python function. In a
builtin that a script calls millions of times the narrowing costs more than the
arithmetic it annotates. This pass rewrites the call to its value operand::

    b = cast(float, base)      ->      b = base
    typing.cast(int, x) + 1    ->      x + 1

An ``if TYPE_CHECKING:`` statement is resolved the same way: the name is False at
runtime, so the statement is replaced by its ``else`` body, or dropped when it
has none::

    if TYPE_CHECKING:                  import json
        import orjson as json   ->
    else:
        import json

It is the one pass that also runs over pynecore's own plain modules (see
``PyneLoader``), because that is where the casts on the hot path are.

A call is erased only when the rewrite provably keeps the module's behaviour:

- the name is bound to ``typing.cast`` / ``typing.TYPE_CHECKING`` by a module-level
  import (``from typing import cast [as name]`` or ``import typing [as name]``) and the
  name is bound NOWHERE else in the module — no assignment, parameter,
  definition, other import, ``except ... as``, pattern capture or ``del`` of it
  at any depth. A module that rebinds the name keeps every call through it;
- the call has exactly two positional operands, no keyword and no star operand;
- the type operand is a plain type expression (names, attributes, subscripts,
  constants, tuples, lists and ``|`` unions of those), which has no effect to
  lose. Such an operand can at most raise ``NameError`` for a name imported
  under ``TYPE_CHECKING`` only; the erased form is more lenient there, never
  different;
- an ``if`` (or ``elif``) is resolved only when its whole test is the trusted name
  (``TYPE_CHECKING`` / ``typing.TYPE_CHECKING``); a compound test is left as written.

The import itself stays: the name may still be used as a value, and an unused
import costs nothing per bar.
"""
import ast

__all__ = ['TypeErasureTransformer', 'erase_type_calls', 'has_erasure_marker']

# Texts one of which a module source must contain for the pass to have anything to
# do; the loader's prefilter, so that an ordinary module is never parsed for nothing
_ERASURE_MARKERS = ('cast(', 'TYPE_CHECKING')
_ERASURE_MARKERS_BYTES = tuple(marker.encode('ascii') for marker in _ERASURE_MARKERS)


def has_erasure_marker(source: bytes | str) -> bool:
    """Whether a module source can contain anything the pass rewrites.

    :param source: The module's source, raw or decoded.
    :return: False when parsing the module for this pass would be wasted work.
    """
    if isinstance(source, bytes):
        return any(marker in source for marker in _ERASURE_MARKERS_BYTES)
    return any(marker in source for marker in _ERASURE_MARKERS)


_TYPE_EXPR_NODES = (ast.Name, ast.Attribute, ast.Subscript, ast.Constant, ast.Tuple,
                    ast.List, ast.BinOp, ast.BitOr, ast.Load)


def _is_type_expression(node: ast.expr) -> bool:
    """Whether evaluating an expression can have no effect worth keeping."""
    for child in ast.walk(node):
        if not isinstance(child, _TYPE_EXPR_NODES):
            return False
        if isinstance(child, ast.BinOp) and not isinstance(child.op, ast.BitOr):
            return False
    return True


class _BindingCounter(ast.NodeVisitor):
    """Count, per name, every construct in a module that binds or unbinds it."""

    def __init__(self) -> None:
        self.bound: dict[str, int] = {}

    def _bind(self, name: str | None) -> None:
        if name is not None:
            self.bound[name] = self.bound.get(name, 0) + 1

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self._bind(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self._bind(node.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name.split('.')[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._bind(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._bind(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self._bind(node.rest)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        # A ``global`` statement lets a function rebind the module name through
        # a plain assignment, which the Name visitor already counts; the
        # declaration itself is counted too so that the name is never trusted
        for name in node.names:
            self._bind(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._bind(name)


class TypeErasureTransformer(ast.NodeTransformer):
    """Replace ``typing.cast(T, x)`` with ``x`` and resolve ``if TYPE_CHECKING:``
    where that is provably the same."""

    def __init__(self) -> None:
        self.erased = 0
        self._cast_names: frozenset[str] = frozenset()
        self._flag_names: frozenset[str] = frozenset()
        self._typing_names: frozenset[str] = frozenset()

    def visit_Module(self, node: ast.Module) -> ast.AST:
        cast_names: dict[str, int] = {}
        flag_names: dict[str, int] = {}
        typing_names: dict[str, int] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == 'typing' and stmt.level == 0:
                for alias in stmt.names:
                    name = alias.asname or alias.name
                    if alias.name == 'cast':
                        cast_names[name] = cast_names.get(name, 0) + 1
                    elif alias.name == 'TYPE_CHECKING':
                        flag_names[name] = flag_names.get(name, 0) + 1
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == 'typing':
                        name = alias.asname or alias.name
                        typing_names[name] = typing_names.get(name, 0) + 1
        if not cast_names and not flag_names and not typing_names:
            return node

        counter = _BindingCounter()
        counter.visit(node)
        # Trusted: every binding of the name in the whole module is one of the
        # typing imports counted above
        self._cast_names = frozenset(
            name for name, count in cast_names.items() if counter.bound.get(name) == count)
        self._flag_names = frozenset(
            name for name, count in flag_names.items() if counter.bound.get(name) == count)
        self._typing_names = frozenset(
            name for name, count in typing_names.items() if counter.bound.get(name) == count)
        if not self._cast_names and not self._flag_names and not self._typing_names:
            return node
        return self.generic_visit(node)

    def _is_cast(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self._cast_names
        return (isinstance(func, ast.Attribute) and func.attr == 'cast'
                and isinstance(func.value, ast.Name) and func.value.id in self._typing_names)

    def _is_flag(self, test: ast.expr) -> bool:
        if isinstance(test, ast.Name):
            return test.id in self._flag_names
        return (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'
                and isinstance(test.value, ast.Name) and test.value.id in self._typing_names)

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        if not self._is_flag(node.test):
            return self.generic_visit(node)
        self.erased += 1
        if not node.orelse:
            # A statement list may not become empty, and the enclosing one is not
            # known here
            return ast.copy_location(ast.Pass(), node)
        kept: list[ast.stmt] = []
        for stmt in node.orelse:
            result = self.visit(stmt)
            if isinstance(result, list):
                kept.extend(result)
            elif result is not None:
                kept.append(result)
        return kept

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (self._is_cast(node.func) and len(node.args) == 2 and not node.keywords
                and not isinstance(node.args[0], ast.Starred)
                and not isinstance(node.args[1], ast.Starred)
                and _is_type_expression(node.args[0])):
            self.erased += 1
            return node.args[1]
        return node


def erase_type_calls(tree: ast.Module) -> ast.Module:
    """Run the erasure over a module tree.

    :param tree: The parsed module.
    :return: The same tree, with the erasable calls replaced by their operand.
    """
    return TypeErasureTransformer().visit(tree)
