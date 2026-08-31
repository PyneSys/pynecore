import ast
from collections.abc import Callable

# The one definition of Pine's comparison tolerance, shared with the measured
# tolerant builtins in ``lib`` (see that module's docstring for the rule and
# its evidence). The boundary belongs to EQUALITY: at a difference of exactly
# ``1e-10`` TradingView still reports the operands equal, hence the strict
# forms below test ``> 1e-10`` / ``< -1e-10`` and the non-strict ones
# ``<= 1e-10`` / ``>= -1e-10``.
from pynecore.core.pine_compare import EPSILON
from .pine_type_rules import BOOL, stamp_lowering

# op -> (bound, comparison of the bound against the difference). ``a < b``
# becomes ``-EPSILON > a - b`` rather than ``b - a > EPSILON``: the difference
# keeps the operands' source evaluation order, and putting the float bound on
# the LEFT saves a dispatch round on integer operands (``int.__lt__(float)``
# returns NotImplemented first, so the reflected float comparison runs anyway
# -- measured 17.0 -> 15.5 ns; float operands are unaffected).
_ORDERING_FORMS: dict[type, tuple[float, type]] = {
    ast.Lt: (-EPSILON, ast.Gt),
    ast.Gt: (EPSILON, ast.Lt),
    ast.LtE: (EPSILON, ast.GtE),
    ast.GtE: (-EPSILON, ast.LtE),
}

# The operators that must hold for two operands that are already equal. Their
# difference form alone cannot decide those: ``inf - inf`` is nan, which makes
# every comparison against the bound false, so the raw comparison is kept in
# front of the tolerant one (see ``_rewrite_pair``).
_NON_STRICT_FORMS = (ast.LtE, ast.GtE)

_TOLERANT_OPS = (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq)

# How many times the rewrite reads each operand of a comparison: the strict
# orderings evaluate a single difference, the non-strict ones a raw comparison
# plus that difference, and ``==``/``!=`` add the type guard and the exact
# branch on top.
_READS: dict[type, int] = {
    ast.Lt: 1, ast.Gt: 1,
    ast.LtE: 2, ast.GtE: 2,
    ast.Eq: 4, ast.NotEq: 3,
}

# The only node kinds whose evaluation provably cannot run user code. The test
# below is inverted -- anything not on this list counts as able to rebind a name
# another operand reads -- because arbitrary Python hides behind an attribute
# (``property``, ``__getattribute__``), a subscript (``__getitem__``) and every
# overloaded operator dunder, not just behind a call.
_PURE_NODES = (ast.Name, ast.Constant, ast.expr_context)


def _unreachable() -> ast.expr:
    """Guard for the reference factory of a single-use operand."""
    raise AssertionError("single-use operand referenced more than once")


def _may_rebind(node: ast.expr) -> bool:
    """Whether evaluating this operand can change what another operand reads."""
    return not all(isinstance(n, _PURE_NODES) for n in ast.walk(node))


def _is_skippable_const(node: ast.expr) -> bool:
    """A ``str``/``bool`` constant operand: Pine only allows comparing it to the
    same type, so the comparison is homogeneous non-float and neither the
    tolerance nor a na guard applies."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (str, bool))


class FloatToleranceTransformer(ast.NodeTransformer):
    """
    Give the comparison operators TradingView's tolerant float semantics.

    Pine does not compare floats bit-exactly: ``<``, ``>``, ``<=``, ``>=``,
    ``==`` and ``!=`` treat operands closer than ``EPSILON`` as equal. Every
    operator is rewritten into an arithmetic form over the difference, with
    the bound kept on the left (see ``_ORDERING_FORMS``)::

        a <  b   ->  -1e-10 >  a - b
        a >  b   ->   1e-10 <  a - b
        a <= b   ->  a <= b or  1e-10 >= a - b
        a >= b   ->  a >= b or -1e-10 <= a - b
        a == b   ->  a == b or -1e-10 <= a - b <= 1e-10
        a != b   ->  1e-10 < (d := a - b) or -1e-10 > d

    The forms are na-correct for free, which is why they are written over the
    difference instead of over ``abs()``: a native nan difference makes every
    one of them False, and an ``NA`` object propagates itself through the
    subtraction into comparisons that are False by definition — exactly Pine,
    where every comparison involving na is false (including ``na != x``).

    The operators that must hold for already-equal operands (``<=``, ``>=``,
    ``==``) keep the raw comparison in front of the tolerant one. The
    difference form alone cannot decide them for two equal infinities, whose
    difference is nan — and ``inf`` is a normal value here, produced by
    ``safe_div`` on a zero denominator with IEEE-754 comparison semantics. The
    raw comparison stays na-correct (nan and ``NA`` compare False) and doubles
    as a short circuit for the ordered case. The strict orderings and ``!=``
    need no such prefix: they are already false for two equal infinities.

    Integers keep exact semantics automatically: an int difference is either 0
    or at least 1, so no tolerance can ever bridge it. Only ``==``/``!=`` are
    guarded by a runtime type test, because they are the operators Pine also
    allows on strings, colors and object references, and those cannot be
    subtracted. The tolerant branch runs only when a real ``float`` is
    involved; everything else keeps the exact comparison, which is already
    correct for ints, bools, strings, objects and non-float ``NA`` instances.

    The guard's second operand is joined with a bitwise ``|`` instead of
    ``or`` whenever it carries a walrus binding: ``or`` would short-circuit and
    leave the temporary unbound for the branches below it.

    Operands that are not simple names/constants are bound once via a walrus so
    their side effects (a call, but also a ``property``, a ``__getitem__`` or an
    operator dunder) don't run twice, and the binding is always emitted at the
    operand's first evaluated position so the source order is preserved. A plain
    name is bound too when a *later* operand can rebind it, because the rewrite
    would otherwise re-read the name after that side effect and see a different
    value than the source expression does.

    Runs on user and compiled scripts only: pynecore's own lib modules
    implement the natively bit-exact builtins (``math.max``, ``ta.crossover``,
    …) and the raw ``x != x`` nan idiom, both of which this rewrite would
    break.
    """

    def __init__(self):
        self._temp_counter = 0

    def _binder(self, node: ast.expr, uses: int,
                volatile: bool) -> tuple[ast.expr, Callable[[], ast.expr]]:
        """Return ``(first_use, ref)`` for an operand.

        Simple names/constants are re-read as fresh nodes (AST nodes must not
        be shared) and an operand the rewrite reads only once stays in place;
        anything else is bound once via a walrus, so ``first_use`` must be
        placed at the operand's first evaluated position.

        :param node: The operand expression.
        :param uses: How many times the rewrite reads the operand.
        :param volatile: Whether a later operand can rebind this one's name.
        """
        if isinstance(node, ast.Constant):
            value = node.value
            return node, lambda: ast.Constant(value=value)
        if isinstance(node, ast.Name) and not (volatile and uses > 1):
            name = node.id
            return node, lambda: ast.Name(id=name, ctx=ast.Load())
        if uses == 1:
            return node, _unreachable
        self._temp_counter += 1
        temp = f"__cmp{self._temp_counter}__"
        return (ast.NamedExpr(target=ast.Name(id=temp, ctx=ast.Store()), value=node),
                lambda: ast.Name(id=temp, ctx=ast.Load()))

    @staticmethod
    def _is_float(operand: ast.expr) -> ast.expr:
        """``operand.__class__ is float`` -- an attribute read instead of a
        ``type()`` call, which measures ~1.2 ns cheaper per operand."""
        return ast.Compare(
            left=ast.Attribute(value=operand, attr='__class__', ctx=ast.Load()),
            ops=[ast.Is()], comparators=[ast.Name(id='float', ctx=ast.Load())])

    def _rewrite_pair(self, op: ast.cmpop, left: ast.expr, left_ref: Callable[[], ast.expr],
                      right: ast.expr, right_ref: Callable[[], ast.expr]) -> ast.expr:
        """Rewrite one ``left op right`` comparison into its tolerant form."""
        form = _ORDERING_FORMS.get(type(op))
        if form is not None:
            bound, cmp_op = form
            if type(op) in _NON_STRICT_FORMS:
                return ast.BoolOp(op=ast.Or(), values=[
                    ast.Compare(left=left, ops=[type(op)()], comparators=[right]),
                    ast.Compare(left=ast.Constant(value=bound), ops=[cmp_op()],
                                comparators=[ast.BinOp(left=left_ref(), op=ast.Sub(),
                                                       right=right_ref())]),
                ])
            return ast.Compare(left=ast.Constant(value=bound), ops=[cmp_op()],
                               comparators=[ast.BinOp(left=left, op=ast.Sub(), right=right)])

        # ``==``/``!=``: the tolerant branch is only reachable for real floats
        guard_left = self._is_float(left)
        guard_right = self._is_float(right)
        guard: ast.expr
        if isinstance(right, ast.NamedExpr):
            guard = ast.BinOp(left=guard_left, op=ast.BitOr(), right=guard_right)
        else:
            guard = ast.BoolOp(op=ast.Or(), values=[guard_left, guard_right])

        if isinstance(op, ast.Eq):
            tolerant: ast.expr = ast.BoolOp(op=ast.Or(), values=[
                ast.Compare(left=left_ref(), ops=[ast.Eq()], comparators=[right_ref()]),
                ast.Compare(
                    left=ast.Constant(value=-EPSILON), ops=[ast.LtE(), ast.LtE()],
                    comparators=[ast.BinOp(left=left_ref(), op=ast.Sub(), right=right_ref()),
                                 ast.Constant(value=EPSILON)]),
            ])
        else:
            self._temp_counter += 1
            diff = f"__cmp{self._temp_counter}__"
            tolerant = ast.BoolOp(op=ast.Or(), values=[
                ast.Compare(
                    left=ast.Constant(value=EPSILON), ops=[ast.Lt()],
                    comparators=[ast.NamedExpr(
                        target=ast.Name(id=diff, ctx=ast.Store()),
                        value=ast.BinOp(left=left_ref(), op=ast.Sub(), right=right_ref()))]),
                ast.Compare(left=ast.Constant(value=-EPSILON), ops=[ast.Gt()],
                            comparators=[ast.Name(id=diff, ctx=ast.Load())]),
            ])
        exact = ast.Compare(left=left_ref(), ops=[type(op)()], comparators=[right_ref()])
        return ast.IfExp(test=guard, body=tolerant, orelse=exact)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        self.generic_visit(node)

        # An earlier pass may have emitted a comparison that already IS the
        # tolerance (``PineTruthinessTransformer``'s bounds): rewriting it would
        # add a second EPSILON to a threshold that was measured on TradingView.
        if getattr(node, 'pine_exact', False):
            return node
        if not all(isinstance(op, _TOLERANT_OPS) for op in node.ops):
            return node
        operands = [node.left, *node.comparators]
        if any(_is_skippable_const(operand) for operand in operands):
            return node

        # In a chain the inner operands are read by both of their clauses
        uses = [0] * len(operands)
        for i, op in enumerate(node.ops):
            reads = _READS[type(op)]
            uses[i] += reads
            uses[i + 1] += reads

        binders = [self._binder(operand, uses[i],
                                any(_may_rebind(later) for later in operands[i + 1:]))
                   for i, operand in enumerate(operands)]
        bound = [False] * len(operands)
        clauses: list[ast.expr] = []
        for i, op in enumerate(node.ops):
            pair: list[ast.expr] = []
            for index in (i, i + 1):
                first, ref = binders[index]
                pair.append(ref() if bound[index] else first)
                bound[index] = True
            clauses.append(self._rewrite_pair(op, pair[0], binders[i][1],
                                              pair[1], binders[i + 1][1]))
        # A chain evaluates the later operands only if the earlier comparisons
        # hold, which is what Python's own chain semantics do
        rewritten = (clauses[0] if len(clauses) == 1
                     else ast.BoolOp(op=ast.And(), values=clauses))
        # Whatever shape the rewrite took, it stands where a comparison stood,
        # so it is a bool; the guards, differences and temporaries it emitted
        # around the preserved operands are typed from those operands
        return stamp_lowering(rewritten, BOOL)
