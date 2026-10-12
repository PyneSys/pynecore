from typing import cast
import ast

from .phase import Rule
from .pine_type_rules import BOOL, FLOAT, INT, STR, TYPELESS, get_ty, set_ty, stamp_lowering

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


class SafeDivisionTransformer(Rule):
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

    With ``pine_mod`` (user code), ``a % b`` becomes ``safe_convert.safe_mod(a, b)``:
    Pine's remainder is truncated (C's ``fmod``) where Python's is floored, and a zero
    divisor gives na. A string left operand is Python's formatting and stays. pynecore's
    own lib modules keep Python's ``%``: their ring buffers index with the floored one.

    An augmented assignment (``x /= y``, ``x %= y``) is written out as the plain
    assignment of the same safe call; a sliced subscript target keeps Python's operator.
    """

    def __init__(self, pine_mod: bool = False):
        self.pine_mod = pine_mod
        self.has_division_operations = False  # Track if division is used
        self._counter = 0
        self._func_depth = 0
        self._blocked = 0
        #: Per enclosing function body, the ``_blocked`` count of the scope around it
        self._outer_blocked: list[int] = []

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

    def _is_pine_mod(self, op: ast.operator, left: ast.expr) -> bool:
        """Whether a ``%`` is Pine's remainder rather than Python's string formatting."""
        return (self.pine_mod and isinstance(op, ast.Mod) and get_ty(left) != STR
                and not (isinstance(left, ast.Constant) and isinstance(left.value, str))
                and not isinstance(left, ast.JoinedStr))

    def _bind_once(self, node: ast.expr, temps: list[str]) -> tuple[ast.expr, ast.expr]:
        """
        Evaluate one part of an augmented assignment's target once for its read and store.

        A literal is read twice. Everything else, a name included, is bound to a temporary
        by the read: the index, the target lookup and the right side all run between the
        read and the store, and any of them may rebind a name.

        :param node: The receiver or the index of the target.
        :param temps: Collects the temporaries bound.
        :return: The expression of the part for the read, and the one for the store.
        """
        if isinstance(node, ast.Constant):
            return node, _reread(node)
        temp = self._temp()
        temps.append(temp)
        return ast.NamedExpr(target=ast.Name(id=temp, ctx=ast.Store()), value=node), _load(temp)

    def leave_AugAssign(self, node: ast.AugAssign) -> ast.stmt | list[ast.stmt]:
        """
        Write out an augmented division or Pine remainder as a plain assignment.

        Python evaluates the receiver and the index of an augmented target once, before
        the right side; the written-out form reads the target through the same values
        it stores to (see :meth:`_bind_once`). A temporary bound outside a function body
        is deleted right after the statement.
        """
        if isinstance(node.op, ast.Div):
            helper = 'safe_div'
        elif self._is_pine_mod(node.op, node.target):
            helper = 'safe_mod'
        else:
            return node
        target = node.target
        temps: list[str] = []
        if isinstance(target, ast.Name):
            current: ast.expr = _load(target.id)
        elif isinstance(target, ast.Attribute):
            receiver, stored = self._bind_once(target.value, temps)
            current = ast.Attribute(value=receiver, attr=target.attr, ctx=ast.Load())
            target = ast.Attribute(value=stored, attr=target.attr, ctx=ast.Store())
        elif isinstance(target, ast.Subscript) and not any(
                isinstance(part, ast.Slice) for part in ast.walk(target.slice)):
            receiver, stored = self._bind_once(target.value, temps)
            index, stored_index = self._bind_once(target.slice, temps)
            current = ast.Subscript(value=receiver, slice=index, ctx=ast.Load())
            target = ast.Subscript(value=stored, slice=stored_index, ctx=ast.Store())
        else:
            return node
        self.has_division_operations = True
        ty = get_ty(node.target)
        value = stamp_lowering(ast.Call(func=_safe_convert_attr(helper),
                                        args=[set_ty(current, ty), node.value], keywords=[]),
                               ty)
        assign = ast.copy_location(ast.Assign(targets=[target], value=value), node)
        if not temps or (self._func_depth and not self._blocked):
            return assign
        return [assign, ast.copy_location(ast.Delete(
            targets=[ast.Name(id=temp, ctx=ast.Del()) for temp in temps]), node)]

    def leave_BinOp(self, node: ast.BinOp) -> ast.expr:
        """
        Transform division and Pine remainder operations
        """
        if self._is_pine_mod(node.op, node.left):
            self.has_division_operations = True
            return ast.copy_location(stamp_lowering(
                ast.Call(func=_safe_convert_attr('safe_mod'), args=[node.left, node.right],
                         keywords=[]), get_ty(node)), node)
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

    def _block(self, node: ast.AST) -> None:
        self._blocked += 1

    def _unblock(self, node: ast.AST) -> ast.AST:
        self._blocked -= 1
        return node

    # A temporary bound in a class body would become a class attribute
    enter_ClassDef = _block
    leave_ClassDef = _unblock
    # A walrus in a lambda binds in the lambda's own scope
    enter_Lambda = _block
    leave_Lambda = _unblock
    # A walrus is illegal in a comprehension's iterable and binds to the
    # containing scope elsewhere in it
    enter_ListComp = enter_SetComp = enter_DictComp = enter_GeneratorExp = _block
    leave_ListComp = leave_SetComp = leave_DictComp = leave_GeneratorExp = _unblock

    def enter_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Decorators and defaults are evaluated in the ENCLOSING scope, so only
        # the body gets the function's own depth. The body is a scope of its
        # own even inside a class body: a temporary bound there is a plain
        # local again
        self._outer_blocked.append(self._blocked)
        self._blocked = 0
        self._func_depth += 1

    def leave_function_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._func_depth -= 1
        self._blocked = self._outer_blocked.pop()

    def leave_Module(self, node: ast.Module) -> ast.Module:
        """
        Add safe_convert import if needed
        """
        # Only add the import if we actually transformed any divisions
        if not self.has_division_operations:
            return node

        # Only an import that binds the module itself to ``safe_convert`` provides
        # the name the emitted calls read
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                module = 'pynecore.core' if stmt.level == 0 else 'core'
                bound = (stmt.module == module and any(
                    alias.name == 'safe_convert' and alias.asname in (None, 'safe_convert')
                    for alias in stmt.names))
            elif isinstance(stmt, ast.Import):
                bound = any(alias.name == 'pynecore.core.safe_convert'
                            and alias.asname == 'safe_convert' for alias in stmt.names)
            else:
                continue
            if bound:
                return node

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
