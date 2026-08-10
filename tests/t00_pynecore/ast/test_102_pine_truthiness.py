"""
Behavior of the tolerant float-to-bool rewrite.

Like the comparison rewrite next to it, the transformer is exercised directly
on expressions: what matters is the semantics of the emitted form. The
reference values come from a TradingView probe run on BINANCE:BTCUSDT 30m,
with every constant multiplied by a runtime ``1.0`` so nothing is folded at
compile time: ``1e-10 ? 1 : 0`` yields 0 and ``1.000001e-10 ? 1 : 0`` yields 1
at both signs, na yields 0, and the ``?:``, ``and``, ``not`` and ``if``
contexts all agree.
"""
import ast

from pynecore.transformers.float_tolerance import FloatToleranceTransformer
from pynecore.transformers.pine_truthiness import PineTruthinessTransformer
from pynecore.types import NA

EPS = 1e-10


def _eval(expr: str, **names):
    """Evaluate an expression through the transform."""
    tree = PineTruthinessTransformer().visit(ast.parse(expr, mode='eval'))
    ast.fix_missing_locations(tree)
    return eval(compile(tree, '<truthiness>', 'eval'), dict(names))  # noqa: S307


def _exec(source: str, **names):
    """Run a statement block through the transform and return its namespace."""
    tree = PineTruthinessTransformer().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    namespace = dict(names)
    exec(compile(tree, '<truthiness>', 'exec'), namespace)  # noqa: S102
    return namespace


def _rewrite(expr: str) -> str:
    """Transformed source of an expression, for shape assertions."""
    tree = PineTruthinessTransformer().visit(ast.parse(expr, mode='eval'))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def __test_float_below_the_tolerance_is_false__():
    """ A float within the tolerance of zero is false, like on TradingView """
    assert _eval("1 if x else 0", x=0.0) == 0
    assert _eval("1 if x else 0", x=2.3684757858670005e-15) == 0
    assert _eval("1 if x else 0", x=-4.5e-14) == 0
    assert _eval("1 if x else 0", x=1e-11) == 0
    assert _eval("1 if x else 0", x=9.9e-11) == 0


def __test_boundary_belongs_to_false__():
    """ Exactly the threshold is still false; one ulp above is true """
    assert _eval("1 if x else 0", x=EPS) == 0
    assert _eval("1 if x else 0", x=-EPS) == 0
    assert _eval("1 if x else 0", x=1.0000000000000002e-10) == 1
    assert _eval("1 if x else 0", x=-1.0000000000000002e-10) == 1
    assert _eval("1 if x else 0", x=1.1e-10) == 1
    assert _eval("1 if x else 0", x=1e-9) == 1


def __test_na_is_false__():
    """ Both na representations are false """
    assert _eval("1 if x else 0", x=float('nan')) == 0
    assert _eval("1 if x else 0", x=NA(int)) == 0


def __test_non_floats_keep_exact_semantics__():
    """ Ints, bools, strings and objects are not put on the tolerance grid """
    assert _eval("1 if x else 0", x=0) == 0
    assert _eval("1 if x else 0", x=1) == 1
    assert _eval("1 if x else 0", x=-1) == 1
    assert _eval("1 if x else 0", x=False) == 0
    assert _eval("1 if x else 0", x=True) == 1
    assert _eval("1 if x else 0", x="") == 0
    assert _eval("1 if x else 0", x="text") == 1
    assert _eval("1 if x else 0", x=object()) == 1


def __test_every_bool_context_converts__():
    """ if / while / and / or / not see the same conversion """
    tiny = 2e-15
    assert _eval("1 if (x and True) else 0", x=tiny) == 0
    assert _eval("1 if (x or False) else 0", x=tiny) == 0
    assert _eval("1 if not x else 0", x=tiny) == 1
    assert _exec("r = 0\nif x:\n    r = 1", x=tiny)['r'] == 0
    assert _exec("r = 0\nif x:\n    r = 1", x=1e-9)['r'] == 1
    assert _exec("r = 0\nwhile x:\n    r = 1\n    x = 0.0", x=tiny)['r'] == 0


def __test_and_or_yield_bools__():
    """ Pine's ``and``/``or`` produce a bool, not one of their operands """
    assert _eval("x and y", x=1.0, y=2.0) is True
    assert _eval("x or y", x=0.0, y=2.0) is True
    assert _eval("x or y", x=0.0, y=1e-15) is False


def __test_operand_is_evaluated_once__():
    """ A non-trivial operand is bound by a walrus, so its side effects run once """
    calls = []

    def f(value):
        calls.append(value)
        return value

    assert _eval("1 if f(0.0) else 0", f=f) == 0
    assert calls == [0.0]


def __test_already_bool_expressions_are_left_alone__():
    """ Comparisons and friends are not wrapped: the conversion would be cost only """
    assert _rewrite("1 if a > b else 0") == "1 if a > b else 0"
    assert _rewrite("1 if a and b > c else 0") == \
        "1 if (-1e-10 > a or 1e-10 < a if a.__class__ is float else a) and b > c else 0"
    assert _rewrite("1 if not (a > b) else 0") == "1 if not a > b else 0"
    assert _rewrite("1 if True else 0") == "1 if True else 0"


def __test_comparison_rewrite_output_is_not_wrapped__():
    """ The ``==``/``!=`` conditional the tolerance rewrite emits is already a bool """
    source = "1 if a == b else 0"
    tree = FloatToleranceTransformer().visit(ast.parse(source, mode='eval'))
    before = ast.unparse(ast.fix_missing_locations(tree))
    tree = PineTruthinessTransformer().visit(tree)
    assert ast.unparse(ast.fix_missing_locations(tree)) == before
