"""
Behavior of the tolerant comparison rewrite.

The transformer is exercised directly on expressions instead of through a
script run: what matters is the semantics of the emitted form, and a plain
``eval`` pins them without an OHLCV feed in the way. The reference values come
from TradingView probes (m542/m545/m546): the tolerance is absolute, its
threshold is exactly the ``1e-10`` double, and the boundary belongs to
equality.
"""
import ast
from math import inf

from pynecore.core.pine_compare import equal
from pynecore.transformers.float_tolerance import FloatToleranceTransformer
from pynecore.types import NA

EPS = 1e-10


def _rewrite(expr: str) -> str:
    """Transformed source of an expression, for shape assertions."""
    tree = FloatToleranceTransformer().visit(ast.parse(expr, mode='eval'))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _eval(expr: str, **names):
    """Evaluate an expression through the transform."""
    tree = FloatToleranceTransformer().visit(ast.parse(expr, mode='eval'))
    ast.fix_missing_locations(tree)
    return eval(compile(tree, '<tolerance>', 'eval'), dict(names))  # noqa: S307


def __test_float_tolerance_ordering__():
    """ Ordering operators treat operands closer than the tolerance as equal """
    # A zero base makes ``b - a`` the exact double, with no rounding grid
    near = dict(a=0.0, b=5e-11)
    assert _eval("a < b", **near) is False
    assert _eval("a > b", **near) is False
    assert _eval("a <= b", **near) is True
    assert _eval("a >= b", **near) is True

    far = dict(a=0.0, b=2e-10)
    assert _eval("a < b", **far) is True
    assert _eval("a > b", **far) is False
    assert _eval("a <= b", **far) is True
    assert _eval("a >= b", **far) is False

    # The same in the negative direction
    assert _eval("a > b", a=0.0, b=-2e-10) is True
    assert _eval("a >= b", a=0.0, b=-2e-10) is True
    assert _eval("a < b", a=0.0, b=-2e-10) is False


def __test_float_tolerance_boundary_belongs_to_equality__():
    """ A difference of exactly the threshold still compares equal """
    at = dict(a=0.0, b=EPS)
    assert _eval("a < b", **at) is False
    assert _eval("a > b", **at) is False
    assert _eval("a <= b", **at) is True
    assert _eval("a >= b", **at) is True
    assert _eval("a == b", **at) is True
    assert _eval("a != b", **at) is False

    # One ulp above the threshold is already a strict inequality
    above = dict(a=0.0, b=1.0000000000000002e-10)
    assert _eval("a < b", **above) is True
    assert _eval("a == b", **above) is False
    assert _eval("a != b", **above) is True


def __test_float_tolerance_scale_independent__():
    """ The tolerance is absolute, so it vanishes against large magnitudes """
    assert _eval("a == b", a=1000.0, b=1000.0 + 5e-11) is True
    assert _eval("a == b", a=1000.0, b=1000.0 + 1e-6) is False
    assert _eval("a < b", a=1000.0, b=1000.0 + 1e-6) is True


def __test_float_tolerance_equality_of_non_floats_is_exact__():
    """ Ints, strings and objects keep exact equality: only a real float
    operand can reach the tolerant branch """
    assert _eval("a == b", a=3, b=3) is True
    assert _eval("a == b", a=3, b=4) is False
    assert _eval("a != b", a=3, b=4) is True
    assert _eval("a == b", a="up", b="up") is True
    assert _eval("a == b", a="up", b="down") is False
    assert _eval("a != b", a="up", b="down") is True

    marker = object()
    assert _eval("a == b", a=marker, b=marker) is True
    assert _eval("a == b", a=marker, b=object()) is False

    # A mixed int/float pair is numeric, so it does get the tolerance
    assert _eval("a == b", a=3, b=3.00000000005) is True


def __test_float_tolerance_ints_stay_exact__():
    """ An integer difference is never small enough to be bridged """
    assert _eval("a < b", a=3, b=4) is True
    assert _eval("a < b", a=4, b=4) is False
    assert _eval("a <= b", a=4, b=4) is True
    assert _eval("a > b", a=4, b=3) is True
    assert _eval("a >= b", a=3, b=4) is False


def __test_float_tolerance_na_is_false_everywhere__():
    """ Every comparison involving na is false, for both na representations """
    for na in (NA(float), NA(int)):
        for expr in ("a < b", "a > b", "a <= b", "a >= b", "a == b", "a != b"):
            assert _eval(expr, a=na, b=1.0) is False, f"{expr} with {na!r} on the left"
            assert _eval(expr, a=1.0, b=na) is False, f"{expr} with {na!r} on the right"
        # na compared with itself is false as well -- including ``!=``
        for expr in ("a == b", "a != b", "a < b", "a >= b"):
            assert _eval(expr, a=na, b=na) is False, f"{expr} with two {na!r}"


def __test_float_tolerance_infinities_keep_ieee_semantics__():
    """ ``safe_div`` returns raw inf/-inf on a zero denominator and documents
    that comparisons on them follow IEEE-754; the tolerance must not change
    that, even though ``inf - inf`` is nan """
    for value in (inf, -inf):
        same = dict(a=value, b=value)
        assert _eval("a == b", **same) is True, f"{value} == itself"
        assert _eval("a <= b", **same) is True, f"{value} <= itself"
        assert _eval("a >= b", **same) is True, f"{value} >= itself"
        assert _eval("a != b", **same) is False, f"{value} != itself"
        assert _eval("a < b", **same) is False, f"{value} < itself"
        assert _eval("a > b", **same) is False, f"{value} > itself"
        assert equal(value, value) is True, f"equal({value}, {value})"

    opposite = dict(a=inf, b=-inf)
    assert _eval("a == b", **opposite) is False
    assert _eval("a > b", **opposite) is True
    assert _eval("a <= b", **opposite) is False
    assert equal(inf, -inf) is False

    # A finite operand still orders against an infinity the IEEE way
    assert _eval("a >= b", a=inf, b=1.0) is True
    assert _eval("a <= b", a=inf, b=1.0) is False
    assert _eval("a <= b", a=-inf, b=1.0) is True


def __test_float_tolerance_operand_is_not_re_read_after_a_side_effect__():
    """ A name the rewrite reads more than once is bound at its first
    evaluation when a later operand can rebind it """
    def run(expr: str, new_a: float, result: float) -> bool:
        """Evaluate in a scope the right operand rebinds while it is running."""
        scope: dict = {}

        def mutate() -> float:
            scope['a'] = new_a
            return result

        scope.update(a=1.0, mutate=mutate)
        tree = FloatToleranceTransformer().visit(ast.parse(expr, mode='eval'))
        ast.fix_missing_locations(tree)
        return eval(compile(tree, '<tolerance>', 'eval'), scope)  # noqa: S307

    # Re-reading ``a`` after ``mutate()`` ran would see 2.0 and report unequal
    assert run("a == mutate()", 2.0, 1.0) is True
    # ...and here it would push the difference past the tolerance
    assert run("a <= mutate()", 1.0 + 1e-9, 1.0 - 1e-11) is True

    # A side-effect-free right operand needs no binding
    assert _rewrite("a == b") == "a == b or -1e-10 <= a - b <= 1e-10 " \
                                 "if a.__class__ is float or b.__class__ is float else a == b"
    assert _rewrite("a <= b") == "a <= b or 1e-10 >= a - b"


def __test_float_tolerance_property_and_subscript_also_rebind__():
    """ A call is not the only operand that runs user code: an attribute read
    goes through ``property``/``__getattribute__`` and a subscript through
    ``__getitem__``, both of which can rebind the name the other operand reads """
    class Mutator:
        """Returns ``result`` and sets ``scope['a']`` to ``new_a`` while doing so."""

        def __init__(self, scope: dict, new_a: float, result: float):
            self._scope, self._new_a, self._result = scope, new_a, result

        def _fire(self) -> float:
            self._scope['a'] = self._new_a
            return self._result

        @property
        def value(self) -> float:
            return self._fire()

        def __getitem__(self, _index: int) -> float:
            return self._fire()

    def run(expr: str, new_a: float, result: float) -> bool:
        scope: dict = {}
        scope.update(a=1.0, obj=Mutator(scope, new_a, result))
        tree = FloatToleranceTransformer().visit(ast.parse(expr, mode='eval'))
        ast.fix_missing_locations(tree)
        return eval(compile(tree, '<tolerance>', 'eval'), scope)  # noqa: S307

    for operand in ("obj.value", "obj[0]"):
        assert run(f"a == {operand}", 2.0, 1.0) is True, operand
        assert run(f"a != {operand}", 2.0, 1.0) is False, operand
        assert run(f"a <= {operand}", 1.0 + 1e-9, 1.0 - 1e-11) is True, operand
        assert run(f"a >= {operand}", 1.0 - 1e-9, 1.0 + 1e-11) is True, operand


def __test_float_tolerance_str_and_bool_constants_are_skipped__():
    """ A str/bool constant operand keeps the comparison untouched: Pine only
    allows comparing it to the same type """
    assert _rewrite("x == 'long'") == "x == 'long'"
    assert _rewrite("x != True") == "x != True"
    assert _rewrite("'long' == x") == "'long' == x"


def __test_float_tolerance_leaves_other_operators_alone__():
    """ Identity and membership are not float comparisons """
    assert _rewrite("x is None") == "x is None"
    assert _rewrite("x in y") == "x in y"
    assert _rewrite("x is not None") == "x is not None"


def __test_float_tolerance_evaluates_operands_once_and_in_order__():
    """ Side-effecting operands are bound once, left to right """
    calls: list[str] = []

    def probe(name, value):
        calls.append(name)
        return value

    assert _eval("f('l', 1.0) < f('r', 2.0)", f=probe) is True
    assert calls == ['l', 'r']

    calls.clear()
    assert _eval("f('l', 1.0) == f('r', 1.0)", f=probe) is True
    assert calls == ['l', 'r']

    calls.clear()
    assert _eval("f('l', 1.0) != f('r', 2.0)", f=probe) is True
    assert calls == ['l', 'r']

    # The exact branch must see the same bound operands
    calls.clear()
    assert _eval("f('l', 'a') == f('r', 'b')", f=probe) is False
    assert calls == ['l', 'r']


def __test_float_tolerance_chained_comparison__():
    """ A chain keeps Python's short-circuit: the middle operand is evaluated
    once, the last one only if the first comparison holds """
    calls: list[str] = []

    def probe(name, value):
        calls.append(name)
        return value

    assert _eval("0.0 < f('m', 5.0) < 10.0", f=probe) is True
    assert calls == ['m']

    calls.clear()
    assert _eval("0.0 < f('m', -5.0) < f('r', 10.0)", f=probe) is False
    assert calls == ['m']  # the right operand is never reached

    # Tolerance applies to every link of the chain
    assert _eval("a < b < c", a=0.0, b=5e-11, c=1.0) is False
