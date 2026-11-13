"""Shape the missing value of a ``request.security`` read after type inference."""
import ast

from . import ast_walk
from .pine_type_rules import (BOOL, FLOAT, NUMERIC, STR, elements_of, get_ty, is_tuple, set_ty,
                              stamp_lowering)


_BOOL_NA_NAME = '__sec_bool_na·__'


class SecurityDefaultTransformer(ast_walk.NodeTransformer):
    """Resolve bool defaults and tuple widths after type inference.

    The bool factory reads the running script's mode, so a library's defaults
    follow its caller just like bool series history and UDT fields do. A read of
    a tuple whose width the security transform could not see gets one ``na`` per
    element. Other defaults stay as the security transform supplied them.
    """

    def __init__(self):
        self._used = False

    def _default(self, default: ast.expr, ty: str) -> ast.expr:
        """Build the missing value of a bool, or of a tuple.

        :param default: The default emitted by the security transform.
        :param ty: The inferred result type.
        :return: The default with bool elements resolved under the runtime mode.
        """
        if ty == BOOL:
            self._used = True
            return ast.copy_location(stamp_lowering(ast.Call(
                func=ast.Name(id=_BOOL_NA_NAME, ctx=ast.Load()),
                args=[], keywords=[]), BOOL), default)
        if is_tuple(ty):
            types = elements_of(ty)
            if isinstance(default, ast.Tuple) and len(default.elts) == len(types):
                defaults = default.elts
                values = [self._default(value, item) for value, item in zip(defaults, types)]
                if all(value is old for value, old in zip(values, defaults)):
                    return default
            elif not isinstance(default, (ast.Tuple, ast.List)):
                # A scalar default of a tuple read: the security transform could not
                # see the width (the expression is a call of a tuple-returning
                # function, and the read stands in a ``return``), the type pass did.
                # One ``na`` per element, or the unpack at the call site raises
                values = [self._element_default(default, item) for item in types]
            else:
                return default
            return ast.copy_location(stamp_lowering(
                ast.Tuple(elts=values, ctx=ast.Load()), ty), default)
        return default

    def _element_default(self, default: ast.expr, ty: str) -> ast.expr:
        """Build the missing value of one element of a tuple read from a scalar default.

        The na of the element's type, as the implicit conversion makes it from a
        typeless ``na`` in a tuple-literal default: the numeric na, the string na,
        the bool na of the runtime mode. Any other element keeps the typeless ``na``.

        :param default: The scalar default emitted by the security transform.
        :param ty: The element's inferred type.
        :return: A new node for the element.
        """
        if ty in NUMERIC:
            return set_ty(ast.copy_location(ast.Attribute(
                value=ast.Name(id='lib', ctx=ast.Load()), attr='_na_float', ctx=ast.Load()),
                default), FLOAT)
        if ty == STR:
            return set_ty(ast.copy_location(ast.Constant(value=''), default), STR)
        return self._default(ast_walk.clone(default), ty)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._used = False
        # Security reads are only ever emitted together with the module's
        # ``__security_contexts__`` registry: without it there is nothing to
        # resolve, and most scripts have none.
        if not any(isinstance(stmt, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '__security_contexts__'
                           for target in stmt.targets)
                   for stmt in node.body):
            return node
        # Each read is resolved on its own: a default never holds another
        # read, so neither the order nor a rebuild of the tree around them
        # matters, and a plain walk finds them all.
        for sub in ast_walk.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == '__sec_read__' and len(sub.args) == 2):
                sub.args[1] = self._default(sub.args[1], get_ty(sub))
        if self._used:
            insert_at = 0
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_at = 1
            while insert_at < len(node.body):
                stmt = node.body[insert_at]
                if not (isinstance(stmt, ast.ImportFrom) and stmt.module == '__future__'):
                    break
                insert_at += 1
            node.body.insert(insert_at, ast.ImportFrom(
                module='pynecore.types.na',
                names=[ast.alias(name='new_bool_na', asname=_BOOL_NA_NAME)], level=0))
        return node
