"""Debugger-safe source-location filling for transformer-built AST nodes.

``ast.fix_missing_locations`` copies the PARENT's full span onto every
location-less node. For a transformer-inserted statement in a function body
the nearest located parent is the ``FunctionDef`` itself, so the statement
inherits the function's whole range (``def`` line .. last body line) — and
CPython derives the bytecode position of attribute/subscript operations from
a span's END, mapping parts of the emitted prologue onto the function's LAST
source line. A breakpoint on that line then fires on every function entry,
mid-prologue, with the body locals still unassigned.

:func:`fix_locations` fills the same holes with single POINT anchors instead:

- a located node is never touched (a missing end position is completed from
  the node's own start, never from the parent's);
- a location-less statement anchors to the earliest source location surviving
  inside it (hoisted expressions keep their original lines), falling back to
  the innermost located ancestor's start — function entry for the prologue;
- every other synthetic node anchors to its enclosing statement's point.

Points, not spans: a synthetic node stamped with a multi-line span would put
its attribute-op line events on the span's end line, re-creating the bug.
"""
import ast

__all__ = ('fix_locations',)


def _located(node: ast.AST) -> bool:
    return getattr(node, 'lineno', None) is not None


def _stamp_point(node: ast.AST, line: int, col: int) -> None:
    node.lineno = node.end_lineno = line  # type: ignore[attr-defined]
    node.col_offset = node.end_col_offset = col  # type: ignore[attr-defined]


def fix_locations(tree: ast.AST, line: int = 1, col: int = 0) -> ast.AST:
    """Fill missing locations in ``tree`` without leaking parent spans.

    Drop-in replacement for :func:`ast.fix_missing_locations` on transformed
    Pyne modules (see module docstring for why the stock helper is unsafe).

    :param tree: Tree to fix in place.
    :param line: Anchor line for top-level location-less nodes.
    :param col: Anchor column for top-level location-less nodes.
    :return: The same tree.
    """
    _fix(tree, line, col)
    return tree


def _fix(node: ast.AST, line: int, col: int) -> None:
    if 'lineno' in node._attributes:
        if _located(node):
            line, col = node.lineno, node.col_offset  # type: ignore[attr-defined]
            if getattr(node, 'end_lineno', None) is None:
                node.end_lineno = line  # type: ignore[attr-defined]
            if getattr(node, 'end_col_offset', None) is None:
                node.end_col_offset = col  # type: ignore[attr-defined]
        else:
            if isinstance(node, ast.stmt):
                # Hoisted payloads keep their source lines — anchor the new
                # statement next to them rather than at the function entry
                inner = [getattr(n, 'lineno') for n in ast.walk(node) if _located(n)]
                if inner:
                    line, col = min(inner), 0
            _stamp_point(node, line, col)
    for child in ast.iter_child_nodes(node):
        _fix(child, line, col)
