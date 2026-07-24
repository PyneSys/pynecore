import ast


def _is_skippable_const(node: ast.expr) -> bool:
    """A ``str``/``bool`` constant operand: Pine only allows comparing it to the
    same type, so the comparison is homogeneous non-float and a na operand is an
    ``NA`` object whose ``__ne__`` is already False — no nan guard needed."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bool))


class NeGuardTransformer(ast.NodeTransformer):
    """
    Give ``!=`` TradingView's na semantics on native nan operands.

    Pine's float na is a native IEEE-754 nan. Raw IEEE keeps ``==``/``<``/``>``/
    ``<=``/``>=`` falsy on nan — matching TradingView, where every comparison
    with na is false — but ``nan != x`` would be True. Compiled scripts keep the
    readable ``(l) != (r)`` form; this transformer rewrites it at load time to

        l == l and r == r and l != r

    which is False whenever either operand is nan. It is semantically neutral
    for every other operand type: ``x == x`` is True for str/int/bool/objects,
    and an ``NA`` object fails its own ``__eq__`` exactly like its ``__ne__``.

    Operands that are not simple names/constants are bound once via a walrus so
    side effects (function calls) don't run twice. ``str``/``bool`` constant
    comparisons are homogeneous non-float in Pine and stay untouched.
    """

    def __init__(self):
        self._temp_counter = 0

    @staticmethod
    def _copy_simple(node: ast.expr) -> ast.expr:
        """Fresh node for re-reading a simple operand (AST nodes must not be shared)."""
        if isinstance(node, ast.Name):
            return ast.Name(id=node.id, ctx=ast.Load())
        assert isinstance(node, ast.Constant)
        return ast.Constant(value=node.value)

    def _bind_once(self, node: ast.expr) -> tuple[ast.expr, ast.expr, ast.expr]:
        """Return (first_use, second_use, third_use) for an operand.

        Simple names/constants are re-read; anything else is bound once via a
        walrus so side effects (function calls) don't run twice."""
        if isinstance(node, (ast.Name, ast.Constant)):
            return node, self._copy_simple(node), self._copy_simple(node)
        self._temp_counter += 1
        name = f"__ne{self._temp_counter}__"
        return (ast.NamedExpr(target=ast.Name(id=name, ctx=ast.Store()), value=node),
                ast.Name(id=name, ctx=ast.Load()),
                ast.Name(id=name, ctx=ast.Load()))

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)

        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.NotEq):
            return node
        left, right = node.left, node.comparators[0]
        if _is_skippable_const(left) or _is_skippable_const(right):
            return node

        left1, left2, left3 = self._bind_once(left)
        right1, right2, right3 = self._bind_once(right)
        return ast.BoolOp(op=ast.And(), values=[
            ast.Compare(left=left1, ops=[ast.Eq()], comparators=[left2]),
            ast.Compare(left=right1, ops=[ast.Eq()], comparators=[right2]),
            ast.Compare(left=left3, ops=[ast.NotEq()], comparators=[right3]),
        ])
