import ast

# The one definition of Pine's comparison tolerance (see ``core/pine_compare``).
# A float is false exactly when it is tolerantly equal to zero, so the bound
# that decides ``x == 0`` decides ``x ? a : b`` as well.
from pynecore.core.pine_compare import EPSILON


def _is_bool_valued(node: ast.expr) -> bool:
    """Whether the expression provably evaluates to a Python bool already.

    Recursing into a conditional matters: the comparison rewrite emits
    ``==``/``!=`` as ``tolerant if <type guard> else exact``, and both of its
    arms are bools, so the conversion around it would be pure cost.

    :param node: The expression in bool context.
    :return: True when no conversion is needed.
    """
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Constant):
        return node.value is None or node.value.__class__ is bool
    if isinstance(node, ast.IfExp):
        return _is_bool_valued(node.body) and _is_bool_valued(node.orelse)
    return False


class PineTruthinessTransformer(ast.NodeTransformer):
    """
    Give the bool contexts TradingView's tolerant float-to-bool conversion.

    Pine converts a float used where a bool is expected by the same rule its
    comparison operators use: the value is true only when it is farther than
    ``EPSILON`` from zero. Python instead treats every non-zero float as true,
    which turns arithmetic residue — the ``2.4e-15`` a stochastic pinned at its
    floor leaves behind, say — into a fired signal TradingView never draws.

    Every bool context is rewritten into the same inline form::

        if x:  ->  if (-1e-10 > x or x > 1e-10) if x.__class__ is float else x:

    An operand that is not a plain name or constant is bound once by a walrus
    inside the type guard, which is the position that evaluates first, so its
    side effects run exactly once and in source order::

        if f(a):  ->  if ((-1e-10 > __bool1__ or __bool1__ > 1e-10)
                           if (__bool1__ := f(a)).__class__ is float
                           else __bool1__):

    The type guard is what keeps the rewrite honest: only a real ``float``
    takes the tolerant branch, so ints stay exact (0 is the only false one),
    the ``NA`` object keeps its own false ``__bool__``, and the types Pine
    never converts — strings, colors, object references — are handed back
    untouched instead of being compared against a number.

    na needs no separate case: a float na is a native nan, and both bounds
    compare false against it, which is exactly Pine's ``na ? a : b`` -> ``b``.

    Expressions that are already bools (comparisons, ``and``/``or``/``not``,
    bool constants, either arm of an already-bool conditional) are left alone.

    ``and`` / ``or`` convert their operands rather than their result: Python's
    operators yield an operand, Pine's yield a bool, so converting both sides
    fixes the value and the truth test in one step.

    Measured on TradingView (BINANCE:BTCUSDT 30m, constants kept off the
    constant-folding path by a runtime factor): ``1e-10`` is false and
    ``1.000001e-10`` is true at both signs, na is false, and the ``?:``,
    ``and``, ``not`` and ``if`` contexts all agree — one rule, one constant.

    Runs on user and compiled scripts only, like the comparison rewrite:
    pynecore's own lib modules implement the builtins natively and rely on
    Python's plain truthiness. It runs early, on the script's own control flow,
    so the ``if`` statements the later passes emit for their own bookkeeping
    (a persistent's lazy-init flag, say) keep their plain test — those are
    bools by construction and would only pay for the guard. The bounds emitted
    here are marked ``pine_exact`` so the comparison rewrite, which runs at the
    end of the pipeline, leaves them at exactly one EPSILON.
    """

    def __init__(self):
        self._temp_counter = 0

    def _convert(self, node: ast.expr) -> ast.expr:
        """Wrap one expression in Pine's float-to-bool conversion.

        :param node: The expression in bool context.
        :return: The converted expression (or the original when it is a bool).
        """
        if _is_bool_valued(node):
            return node

        # A name or a constant is side-effect free, so it may simply be re-read
        # (AST nodes must not be shared, hence the fresh node per reference).
        # Anything else can run user code — arbitrary Python hides behind an
        # attribute, a subscript and every operator dunder, not just behind a
        # call — so it is bound once instead.
        if isinstance(node, ast.Name):
            name = node.id
            guarded: ast.expr = node

            def ref() -> ast.expr:
                return ast.Name(id=name, ctx=ast.Load())
        elif isinstance(node, ast.Constant):
            value = node.value
            guarded = node

            def ref() -> ast.expr:
                return ast.Constant(value=value)
        else:
            self._temp_counter += 1
            temp = f'__bool{self._temp_counter}__'
            guarded = ast.NamedExpr(target=ast.Name(id=temp, ctx=ast.Store()), value=node)

            def ref() -> ast.expr:
                return ast.Name(id=temp, ctx=ast.Load())

        below = ast.Compare(left=ast.Constant(value=-EPSILON), ops=[ast.Gt()],
                            comparators=[ref()])
        above = ast.Compare(left=ast.Constant(value=EPSILON), ops=[ast.Lt()],
                            comparators=[ref()])
        # The bounds ARE the tolerance; running the comparison rewrite over them
        # would widen each one by another EPSILON and move the threshold to 2e-10
        below.pine_exact = True  # type: ignore[attr-defined]
        above.pine_exact = True  # type: ignore[attr-defined]
        # The non-float arm is truth-tested too, not handed back raw: Pine's
        # ``and``/``or`` yield a bool while Python's yield an operand, so an int
        # or an ``NA`` operand would otherwise leak out as the whole
        # expression's value (a marker plot exports na where TradingView
        # exports 0). Two ``not``s are the cheapest bool cast there is.
        return ast.IfExp(
            test=ast.Compare(
                left=ast.Attribute(value=guarded, attr='__class__', ctx=ast.Load()),
                ops=[ast.Is()], comparators=[ast.Name(id='float', ctx=ast.Load())]),
            body=ast.BoolOp(op=ast.Or(), values=[below, above]),
            orelse=ast.UnaryOp(op=ast.Not(), operand=ast.UnaryOp(op=ast.Not(), operand=ref())))

    def visit_If(self, node: ast.If) -> ast.If:
        self.generic_visit(node)
        node.test = self._convert(node.test)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.test = self._convert(node.test)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.IfExp:
        self.generic_visit(node)
        node.test = self._convert(node.test)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        self.generic_visit(node)
        node.values = [self._convert(value) for value in node.values]
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            node.operand = self._convert(node.operand)
        return node
