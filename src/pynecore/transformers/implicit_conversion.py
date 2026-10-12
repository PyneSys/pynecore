"""
Pine's implicit conversions of a value where it goes, which the type pass marks
(``WIDEN_ATTR``, see ``pine_type_infer``).

An ``array<float>``, ``matrix<float>`` or ``map<K, float>`` holds floats: Pine
converts an int that goes in, so the stored element reads back as ``1.0``, prints
as ``1.0`` in ``str.format`` and makes a float-array overload match. A Python list
stores whatever it is handed, and an int literal stays a Python int. This pass
converts every value the type pass marked as such an int:

- one element (``array.push``, ``array.set``, ``map.put``, ``matrix.set``, the
  initial value of ``array.new`` / ``matrix.new`` and the items of
  ``array.from`` building a float container): an int literal becomes a float
  literal, any other value is wrapped in ``float(...)`` -- the safe conversion
  pass turns that into the na-safe ``safe_float``;
- the elements of an int array or matrix copied in (``array.concat``,
  ``matrix.add_row`` / ``add_col``, ``matrix.concat``): the operand is wrapped in
  ``safe_convert.safe_float_items`` / ``safe_float_matrix``, a converted copy.

A typeless ``na`` literal that goes where a string goes -- a string variable or
field, a string parameter, a ternary arm next to a string, an element of a string
container, the key of a string-keyed map, a string parameter of a lib function -- is
the string na, which TradingView does not tell from ``""``: it becomes the empty string
literal. Passed as compile-time text of the study (a plot title, a ``plotchar`` char,
an ``alertcondition`` message, see ``NA_TEXT_PARAMS``) it is the text ``na`` instead.

A generic body is shared by every context it is instantiated in, and one value in it
can need a conversion in one context and none in another: ``array.push(a, v)`` into a
float array from one call site and into an int array from another. Such a value is
wrapped in the call the type pass prepared for it (``get_widen_site``),
``safe_convert.implicit_convert(value, None)``. That call is an instance-varying site:
the function isolation pass replaces the None with the entry of the instance vector
that belongs to it, so each instance converts the value the way its own context
needs.
"""
from typing import cast
import ast

from . import ast_walk
from .pine_type_rules import (
    FLOAT, STR, WIDEN_FLOAT, WIDEN_ITEMS, WIDEN_MATRIX, WIDEN_NA_TEXT, WIDEN_STRING_NA, array_of,
    get_ty,
    get_widen, get_widen_site, matrix_of, set_ty, set_widen, set_widen_site, stamp_lowering,
)

__all__ = ['ImplicitConversionTransformer']

#: The conversion helper a copied-in container is wrapped in, and the type it returns
_COPY_HELPERS = {
    WIDEN_ITEMS: ('safe_float_items', array_of(FLOAT)),
    WIDEN_MATRIX: ('safe_float_matrix', matrix_of(FLOAT)),
}


class ImplicitConversionTransformer(ast_walk.NodeTransformer):
    """Convert the values the type pass marked where Pine converts them implicitly."""

    def __init__(self) -> None:
        self._needs_import = False

    def visit(self, node: ast.AST) -> ast.AST:
        node = super().visit(node)
        if not isinstance(node, ast.expr):
            return node
        site = get_widen_site(node)
        if site is not None:
            # Wrapped once, like a conversion: a clone must not wrap it again
            set_widen_site(node, None)
            site.args[0] = node
            self._needs_import = True
            return stamp_lowering(site, get_ty(node))
        kind = get_widen(node)
        if kind is None:
            return node
        # Converted once: a clone of this node (a security slice) must not convert again
        set_widen(node, None)
        if kind == WIDEN_STRING_NA:
            return set_ty(ast.copy_location(ast.Constant(value=''), node), STR)
        if kind == WIDEN_NA_TEXT:
            return set_ty(ast.copy_location(ast.Constant(value='na'), node), STR)
        if kind == WIDEN_FLOAT:
            literal = _int_literal(node)
            if literal is not None:
                return set_ty(ast.copy_location(ast.Constant(value=float(literal)), node), FLOAT)
            return stamp_lowering(ast.copy_location(ast.Call(
                func=ast.Name(id='float', ctx=ast.Load()), args=[node], keywords=[]), node), FLOAT)
        helper, result = _COPY_HELPERS[kind]
        self._needs_import = True
        return stamp_lowering(ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id='safe_convert', ctx=ast.Load()), attr=helper,
                               ctx=ast.Load()),
            args=[node], keywords=[]), node), result)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node = cast(ast.Module, self.generic_visit(node))
        if self._needs_import and not _binds_safe_convert(node):
            position = 0
            if node.body and isinstance(node.body[0], ast.Expr) \
                    and isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant):
                position = 1
            while position < len(node.body) \
                    and isinstance(node.body[position], (ast.Import, ast.ImportFrom)):
                position += 1
            node.body.insert(position, ast.ImportFrom(
                module='pynecore.core', names=[ast.alias(name='safe_convert')], level=0))
        return node


def _int_literal(node: ast.expr) -> int | None:
    """The value of an int literal, a signed one included (``-3`` is a ``UnaryOp``).

    :param node: An expression
    :return: The int it spells, or None when it is not an int literal
    """
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return sign * node.value
    return None


def _binds_safe_convert(module: ast.Module) -> bool:
    """Whether the module already binds the ``safe_convert`` module name at its top level."""
    for stmt in module.body:
        if isinstance(stmt, ast.ImportFrom):
            name = (stmt.module or '').removeprefix('pynecore.')
            if name == 'core.safe_convert' and any(alias.asname == 'safe_convert'
                                                   for alias in stmt.names):
                return True
            if name == 'core' and any((alias.asname or alias.name) == 'safe_convert'
                                      for alias in stmt.names):
                return True
        elif isinstance(stmt, ast.Import):
            if any(alias.name == 'pynecore.core.safe_convert' and alias.asname == 'safe_convert'
                   for alias in stmt.names):
                return True
    return False
