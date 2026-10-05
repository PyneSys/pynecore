"""Preserve boolean types in missing ``request.security`` results."""
import ast

from . import ast_walk
from .pine_type_rules import BOOL, elements_of, get_ty, is_tuple, stamp_lowering


_BOOL_NA_NAME = '__sec_bool_na·__'


class SecurityDefaultTransformer(ast_walk.NodeTransformer):
    """Resolve bool defaults after type inference, including tuple elements.

    The bool factory reads the running script's mode, so a library's defaults
    follow its caller just like bool series history and UDT fields do. Other
    result types retain the defaults the security transform supplied.
    """

    def __init__(self):
        self._used = False

    def _default(self, default: ast.expr, ty: str) -> ast.expr:
        """Build the missing value of a bool, or of a tuple containing bools.

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
            else:
                defaults = [default for _ in types]
            values = [self._default(value, item) for value, item in zip(defaults, types)]
            if any(value is not old for value, old in zip(values, defaults)):
                return ast.copy_location(stamp_lowering(
                    ast.Tuple(elts=values, ctx=ast.Load()), ty), default)
        return default

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if (isinstance(node.func, ast.Name) and node.func.id == '__sec_read__'
                and len(node.args) == 2):
            node.args[1] = self._default(node.args[1], get_ty(node))
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._used = False
        node = self.generic_visit(node)  # type: ignore[assignment]
        if self._used:
            insert_at = 0
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                insert_at = 1
            while (insert_at < len(node.body)
                   and isinstance(node.body[insert_at], ast.ImportFrom)
                   and node.body[insert_at].module == '__future__'):
                insert_at += 1
            node.body.insert(insert_at, ast.ImportFrom(
                module='pynecore.types.na',
                names=[ast.alias(name='new_bool_na', asname=_BOOL_NA_NAME)], level=0))
        return node
