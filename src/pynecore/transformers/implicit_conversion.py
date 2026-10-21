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
literal. One that goes where a float or an int goes -- a numeric variable, field,
parameter or declared return, a ternary arm or an arithmetic operand next to a number,
an element of a numeric container -- is the numeric na: it becomes ``lib._na_float``,
the NaN that ``NA(float)`` and ``NA(int)`` both are, so a container, an overload
dispatch or a library receives the na the way Pine typed it. One that goes where an
object of a class goes -- a UDT or drawing variable, field, parameter or declared
return, a ternary arm next to such an object, an element of a container of them -- is
the na OF that class: it becomes ``lib.na(<class>)``, the class spelled the way the
module names it (``get_na_class``), so an overload dispatch on the value sees the
class Pine typed it with and not a typeless na that answers every candidate. A field
default runs while its class body does, so where the class it names is not bound yet
-- the class itself, one the module defines later -- the default is deferred to the
creation of the object, ``field(default_factory=lambda: lib.na(<class>))``. One that
goes where an array, a map or a matrix goes -- a container variable, field, parameter
or declared return, a ternary arm next to a container, a constructor argument for such
a field -- is the na of that container, ``lib.na(list)``, ``lib.na(dict)`` or
``lib.na(<Matrix>)``, the way a compiled container declaration spells it. Passed as
compile-time text of the study (a plot title, a ``plotchar`` char, an
``alertcondition`` message, see ``NA_TEXT_PARAMS``) it is the text ``na`` instead.

A signed ``-na`` is the typeless ``na`` too, and is replaced the same way. A value
that holds it without spelling it -- a parameter defaulted to it, a variable -- is
converted the same way, at run time: it is wrapped in
``safe_convert.implicit_convert(value, kind)``, which turns the typeless na into the
na it goes in as and passes any other value through, so a call site that hands the
function a value of its own gets that value back.

A generic body is shared by every context it is instantiated in, and one value in it
can need a conversion in one context and none in another: ``array.push(a, v)`` into a
float array from one call site and into an int array from another. Such a value is
wrapped in the call the type pass prepared for it (``get_widen_site``),
``safe_convert.implicit_convert(value, None)``. That call is an instance-varying site:
the function isolation pass replaces the None with the entry of the instance vector
that belongs to it, so each instance converts the value the way its own context
needs.
"""
from collections import Counter
from typing import cast
import ast

from . import ast_walk
from .pine_type_rules import (
    CONTAINER_NA_WIDENS, FLOAT, OBJECT, STR, WIDEN_FLOAT, WIDEN_ITEMS, WIDEN_MATRIX,
    WIDEN_NA_TEXT, WIDEN_NUMBER_NA,
    WIDEN_STRING_NA, array_of, class_of, get_na_class,
    get_ty,
    get_widen, get_widen_site, is_na_literal, is_na_widen, matrix_of, set_ty, set_widen,
    set_widen_site, stamp_lowering,
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
        #: The classes the module defines at its top level
        self._module_classes: set[str] = set()
        #: The module's classes whose definition has run before the statement visited
        self._defined_classes: set[str] = set()
        #: The direct field defaults of the class bodies being visited, by node id
        self._field_defaults: set[int] = set()
        #: The name ``dataclasses.field`` is called by in a deferred field default
        self._field_alias: str | None = None
        #: Whether the module has to import ``dataclasses.field`` under that name
        self._needs_field_import = False
        #: How many times the module binds each name, imports and definitions included
        self._bindings: Counter[str] = Counter()
        #: The names a ``from dataclasses import field`` ahead of every class binds
        self._field_imports: set[str] = set()

    def visit(self, node: ast.AST) -> ast.AST:
        original = node
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
        if is_na_widen(kind) and not is_na_literal(node):
            self._needs_import = True
            if class_of(kind) is not None:
                # The na of a class: the conversion is handed the class itself
                converted, result = _class_expr(node), kind
            else:
                converted = ast.Constant(value=kind)
                result = FLOAT if kind == WIDEN_NUMBER_NA \
                    else OBJECT if kind in CONTAINER_NA_WIDENS else STR
            converted_value = stamp_lowering(ast.copy_location(ast.Call(
                func=ast.Attribute(value=ast.Name(id='safe_convert', ctx=ast.Load()),
                                   attr='implicit_convert', ctx=ast.Load()),
                args=[node, converted], keywords=[]), node), result)
            return self._field_default(original, converted_value, result)
        if class_of(kind) is not None or kind in CONTAINER_NA_WIDENS:
            # The na of a class or of a container: ``lib.na(<class>)``
            result = OBJECT if kind in CONTAINER_NA_WIDENS else kind
            return self._field_default(original, stamp_lowering(ast.copy_location(ast.Call(
                func=ast.Attribute(value=ast.Name(id='lib', ctx=ast.Load()), attr='na',
                                   ctx=ast.Load()),
                args=[_class_expr(node)], keywords=[]), node), result), result)
        if kind == WIDEN_STRING_NA:
            return set_ty(ast.copy_location(ast.Constant(value=''), node), STR)
        if kind == WIDEN_NUMBER_NA:
            return set_ty(ast.copy_location(ast.Attribute(
                value=ast.Name(id='lib', ctx=ast.Load()), attr='_na_float', ctx=ast.Load()),
                node), FLOAT)
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

    def _field_default(self, original: ast.expr, value: ast.expr, result: str) -> ast.expr:
        """
        Defer a field default that names a class its class body cannot reach yet.

        A class's body runs before the class is bound, and before every class the
        module defines after it: a default naming one of those raises NameError
        where the body runs. The dataclass builds such a default for each object
        it creates instead, from a factory that names the class once it is bound.

        :param original: The default as the type pass stamped it
        :param value: Its conversion into the na of a class
        :param result: The Pine type of the conversion
        :return: ``value``, or the deferred field built from it
        """
        spelled = get_na_class(original)
        if id(original) not in self._field_defaults or spelled not in self._module_classes \
                or spelled in self._defined_classes:
            return value
        if self._field_alias is None:
            self._field_alias = self._dataclass_field_alias()
        factory = ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                kw_defaults=[], defaults=[]), body=value)
        return stamp_lowering(ast.copy_location(ast.Call(
            func=ast.Name(id=self._field_alias, ctx=ast.Load()), args=[],
            keywords=[ast.keyword(arg='default_factory', value=ast.copy_location(
                factory, original))]), original), result)

    def _dataclass_field_alias(self) -> str:
        """
        The name a deferred field default calls ``dataclasses.field`` by.

        A top-level ``from dataclasses import field`` the module binds no other way
        is reused; otherwise the name is imported under an alias no binding of the
        module takes, the way the compiler names it.

        :return: The name, imported by the module when ``_needs_field_import`` is set
        """
        alias = '_dataclass_field'
        while True:
            if alias in self._field_imports and self._bindings[alias] == 1:
                return alias
            if not self._bindings[alias]:
                self._needs_field_import = True
                return alias
            alias += '_'

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        fields = {id(stmt.value) for stmt in node.body
                  if isinstance(stmt, ast.AnnAssign) and stmt.value is not None}
        self._field_defaults |= fields
        try:
            node = cast(ast.ClassDef, self.generic_visit(node))
        finally:
            self._field_defaults -= fields
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                self._module_classes.add(stmt.name)
            elif isinstance(stmt, ast.ImportFrom) and stmt.module == 'dataclasses' \
                    and not stmt.level and not self._module_classes:
                self._field_imports.update(alias.asname or alias.name for alias in stmt.names
                                           if alias.name == 'field')
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Load):
                self._bindings[child.id] += 1
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._bindings[child.name] += 1
            elif isinstance(child, ast.alias):
                self._bindings[(child.asname or child.name).split('.')[0]] += 1
        body: list[ast.stmt] = []
        for stmt in node.body:
            body.append(cast(ast.stmt, self.visit(stmt)))
            if isinstance(stmt, ast.ClassDef):
                self._defined_classes.add(stmt.name)
        node.body = body
        if self._needs_import and not _binds_safe_convert(node):
            node.body.insert(_import_position(node), ast.ImportFrom(
                module='pynecore.core', names=[ast.alias(name='safe_convert')], level=0))
        if self._needs_field_import:
            node.body.insert(_import_position(node), ast.ImportFrom(
                module='dataclasses', names=[ast.alias(name='field', asname=self._field_alias)],
                level=0))
        return node


def _import_position(module: ast.Module) -> int:
    """
    Where a generated import goes: after the docstring and the module's own imports.

    :param module: The module
    :return: The index of the statement the import is inserted in front of
    """
    position = 0
    if module.body and isinstance(module.body[0], ast.Expr) \
            and isinstance(cast(ast.Expr, module.body[0]).value, ast.Constant):
        position = 1
    while position < len(module.body) \
            and isinstance(module.body[position], (ast.Import, ast.ImportFrom)):
        position += 1
    return position


def _class_expr(node: ast.expr) -> ast.expr:
    """
    The expression naming the class whose na a typeless ``na`` becomes.

    :param node: The na value, stamped with the class's spelling (``get_na_class``)
    :return: The spelling as a name or attribute chain, at the value's location
    """
    head, *attrs = (get_na_class(node) or '').split('.')
    expr: ast.expr = ast.Name(id=head, ctx=ast.Load())
    for attr in attrs:
        expr = ast.Attribute(value=expr, attr=attr, ctx=ast.Load())
    return ast.copy_location(expr, node)


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
