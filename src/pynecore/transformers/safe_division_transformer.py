from typing import cast
import ast

from . import ast_walk
from .pine_type_rules import BOOL, FLOAT, INT, TYPELESS, get_ty, set_ty, stamp_lowering

#: Operand types whose runtime values are native numbers, bools or na (an ``NA``
#: object or a nan). Division on those can raise nothing but
#: ``ZeroDivisionError``, which the inline form keeps away from its fast path.
_INLINE_TYPES = frozenset({INT, FLOAT, BOOL, TYPELESS})

#: Temporary of an inlined division. The middle dot cannot appear in a Python
#: identifier, so neither a script variable nor another pass's temporary can
#: collide with it.
_TEMP_NAME = '__div{}·__'


def _is_numeric_constant(node: ast.expr) -> bool:
    """Whether a node is an int or float literal (a bool is not one)."""
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _safe_convert_attr(attr: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id='safe_convert', ctx=ast.Load()), attr=attr,
                         ctx=ast.Load())


def _load(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def _reread(node: ast.expr) -> ast.expr:
    """A fresh read of a name or literal operand (AST nodes must not be shared)."""
    if isinstance(node, ast.Name):
        return _load(node.id)
    return ast.Constant(value=cast(ast.Constant, node).value)


class SafeDivisionTransformer(ast_walk.NodeTransformer):
    """
    Transformer that converts division operations to safe alternatives
    that preserve Pine Script semantics.

    Every division whose operands are not both literals gets Pine's semantics
    from :func:`pynecore.core.safe_convert.safe_div`: division by zero answers
    inf/-inf/nan instead of raising, and a na operand answers na.

    Inside a function body, a division of numeric-typed operands is written out
    as an expression that runs the plain division first and calls ``safe_div``
    only when that cannot be its result::

        a / b  ->  R if (R := a / (b or safe_convert.zero_divisor)) == R
                   else safe_convert.safe_div(a, b)

    A quotient that equals itself is exactly what ``safe_div`` returns: a na
    operand (a nan, or an ``NA`` object, whose arithmetic yields itself) always
    produces a quotient that fails ``==``. A zero (falsy) divisor is swapped for
    ``zero_divisor``, whose reflected division answers nan, so it lands in the
    fallback too without raising. The fallback gets the operands already
    evaluated, so ``safe_div`` decides every other case -- the zero-divisor
    signs, the interned na, ``inf / inf`` -- with its own code. A literal
    nonzero divisor needs no swap, and a literal zero one keeps the plain call.

    Operands are evaluated exactly once and in source order: a literal is
    re-read, a name is re-read unless the divisor could rebind it first, and
    everything else is bound once with an assignment expression. The inline
    form is only used where such a binding is a plain function local, so module
    level, class bodies, lambdas, comprehensions, decorators and default
    arguments keep the call.

    The inline form requires both operands to be numeric-typed by the type pass:
    for those, the plain division can raise nothing but ``ZeroDivisionError``,
    while ``safe_div`` also turns a ``TypeError`` into na.
    """

    def __init__(self):
        self.has_safe_convert_import = False
        self.has_division_operations = False  # Track if division is used
        self._counter = 0
        self._func_depth = 0
        self._blocked = 0

    def _temp(self) -> str:
        self._counter += 1
        return _TEMP_NAME.format(self._counter)

    def _inline(self, node: ast.BinOp) -> ast.expr | None:
        """
        Write one division out as the fast-path expression.

        :param node: The division, operands already transformed.
        :return: The expression, or None when the division keeps its call.
        """
        left, right = node.left, node.right
        if get_ty(left) not in _INLINE_TYPES or get_ty(right) not in _INLINE_TYPES:
            return None
        right_const = _is_numeric_constant(right)
        if isinstance(right, ast.Constant) and not right_const:
            return None
        if right_const:
            value = cast(ast.Constant, right).value
            # A zero, nan or inf literal divisor is not worth a fast path
            if not value or value != value or value in (float('inf'), float('-inf')):
                return None

        # The left operand is re-read by the fallback, AFTER the right one ran:
        # only a literal, or a name no right operand can rebind, may be re-read
        if isinstance(left, ast.Constant) or (
                isinstance(left, ast.Name)
                and isinstance(right, (ast.Name, ast.Constant))):
            left_first: ast.expr = left

            def left_ref() -> ast.expr:
                return _reread(left)
        else:
            left_temp = self._temp()
            left_first = ast.NamedExpr(target=ast.Name(id=left_temp, ctx=ast.Store()),
                                       value=left)

            def left_ref() -> ast.expr:
                return _load(left_temp)

        if isinstance(right, (ast.Name, ast.Constant)):
            right_first: ast.expr = right

            def right_ref() -> ast.expr:
                return _reread(right)
        else:
            right_temp = self._temp()
            right_first = ast.NamedExpr(target=ast.Name(id=right_temp, ctx=ast.Store()),
                                        value=right)

            def right_ref() -> ast.expr:
                return _load(right_temp)

        divisor = right_first if right_const else ast.BoolOp(
            op=ast.Or(), values=[right_first, _safe_convert_attr('zero_divisor')])
        # Every node that carries the quotient carries the division's type,
        # the fallback call included
        ty = get_ty(node)
        result = self._temp()
        test = ast.Compare(
            left=set_ty(ast.NamedExpr(
                target=ast.Name(id=result, ctx=ast.Store()),
                value=set_ty(ast.BinOp(left=left_first, op=ast.Div(), right=divisor), ty)), ty),
            ops=[ast.Eq()], comparators=[set_ty(_load(result), ty)])
        # A raw self-equality test, not a Pine comparison: the tolerance
        # rewrite must leave it alone
        setattr(test, 'pine_exact', True)
        return ast.IfExp(
            test=test, body=set_ty(_load(result), ty),
            orelse=set_ty(ast.Call(func=_safe_convert_attr('safe_div'),
                                   args=[left_ref(), right_ref()], keywords=[]), ty))

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        """
        Visit BinOp nodes and transform division operations
        """
        # Continue normal transformation for children
        self.generic_visit(node)

        # Check if it's a division operation
        if isinstance(node.op, ast.Div):
            # A literal division by a nonzero literal (e.g. 1/2) cannot raise
            # and stays as is for performance; a literal zero divisor is left
            # unfolded by the constant folder, so it still needs Pine semantics
            if not (isinstance(node.left, ast.Constant) and _is_numeric_constant(node.right)
                    and cast(ast.Constant, node.right).value):
                # Mark that we need the safe_convert import
                self.has_division_operations = True

                expression = None
                if self._func_depth and not self._blocked:
                    expression = self._inline(node)
                if expression is None:
                    expression = ast.Call(func=_safe_convert_attr('safe_div'),
                                          args=[node.left, node.right], keywords=[])
                # The replacement takes over the division's Pine type --
                # ``int / int`` is int-typed, and this is the node the
                # overload pin lands on
                return ast.copy_location(stamp_lowering(expression, get_ty(node)), node)

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
        node.decorator_list = [cast(ast.expr, self.visit(d)) for d in node.decorator_list]
        node.args = cast(ast.arguments, self.visit(node.args))
        if node.returns is not None:
            node.returns = cast(ast.expr, self.visit(node.returns))
        # The body is a scope of its own even inside a class body: a temporary
        # bound there is a plain local again
        blocked, self._blocked = self._blocked, 0
        self._func_depth += 1
        node.body = [cast(ast.stmt, self.visit(stmt)) for stmt in node.body]
        self._func_depth -= 1
        self._blocked = blocked
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

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Add safe_convert import if needed
        """
        # Process the module first
        node = cast(ast.Module, self.generic_visit(node))

        # Only add the import if we actually transformed any divisions
        if not self.has_division_operations:
            return node

        # Check for existing safe_convert import
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == 'pynecore.core.safe_convert':
                self.has_safe_convert_import = True
                # Check if it's imported as 'safe_convert'
                for alias in stmt.names:
                    if alias.name == 'safe_convert' or alias.asname == 'safe_convert':
                        return node
            elif isinstance(stmt, ast.ImportFrom) and stmt.module == 'pynecore.core':
                for alias in stmt.names:
                    if alias.name == 'safe_convert':
                        self.has_safe_convert_import = True
                        return node

        # Add import if needed
        if not self.has_safe_convert_import:
            import_stmt = ast.ImportFrom(
                module='pynecore.core',
                names=[ast.alias(name='safe_convert', asname=None)],
                level=0
            )

            # Find the right position to insert import - after the docstring if it exists
            insert_pos = 0
            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant)):
                insert_pos = 1

            # Insert after any existing imports
            while (insert_pos < len(node.body) and
                   (isinstance(node.body[insert_pos], ast.Import) or
                    isinstance(node.body[insert_pos], ast.ImportFrom))):
                insert_pos += 1

            node.body.insert(insert_pos, import_stmt)

        return node
