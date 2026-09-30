"""
The inline division emission and ``safe_div`` against the Pine division semantics.

Not a Pyne module on purpose: the reference division below must stay a plain
Python ``/``, which the import hook would otherwise lower through the very pass
under test.
"""
import ast
import struct
from math import inf
from typing import Any

from pynecore.core.safe_convert import safe_div, zero_divisor
from pynecore.transformers.float_tolerance import FloatToleranceTransformer
from pynecore.transformers.pine_type_rules import FLOAT, set_ty
from pynecore.transformers.safe_division_transformer import SafeDivisionTransformer
from pynecore.types.na import NA, na_float, na_int


def __test_helper_reference_div(a: Any, b: Any) -> Any:
    """The division semantics every emitted form must reproduce.

    :param a: The numerator.
    :param b: The denominator.
    :return: The Pine quotient.
    """
    if not (a == a) or not (b == b):
        return na_float
    try:
        return a / b
    except ZeroDivisionError:
        if a > 0:
            return inf
        if a < 0:
            return -inf
        return na_float
    except TypeError:
        return na_float


def __test_helper_key(fn: Any, *args: Any) -> Any:
    """A comparison key of one call's outcome that ``==`` alone cannot give.

    The interned ``na_float`` is told apart from a computed nan, a float is
    keyed by its bit pattern (``-0.0`` differs from ``0.0``), and a raised
    exception by its type.

    :param fn: The callable to run.
    :param args: Its arguments.
    :return: A hashable key.
    """
    try:
        value = fn(*args)
    except Exception as exc:  # noqa: the exception TYPE is the compared result
        return 'raises', type(exc)
    if value is na_float:
        return 'na_float',
    if isinstance(value, NA):
        return 'NA', value.type
    if type(value) is float:
        return 'f', struct.pack('<d', value)
    return type(value).__name__, value


#: Values in the domain of a numeric-typed Pine operand
__test_helper_VALUES = (
    0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 3.0, 1e-300, 1e300, -1e300, 5e-324,
    inf, -inf, float('nan'), na_float, na_int,
    0, 1, -1, 7, True, False,
    NA(None), NA(float), NA(int), NA(bool),
)

#: Values only an untyped operand can carry
__test_helper_FOREIGN = (None, 'x', 10 ** 400, -10 ** 400)


class _DivisionStamper(ast.NodeVisitor):
    """Types every division operand as float, standing in for the type pass."""

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            set_ty(node.left, FLOAT)
            set_ty(node.right, FLOAT)
            set_ty(node, FLOAT)


def __test_helper_compile(source: str, typed: bool = True) -> dict[str, Any]:
    """Run the division pass (and the tolerance pass after it) over a module.

    :param source: The module source.
    :param typed: Whether the division operands carry a numeric type.
    :return: The executed module namespace.
    """
    tree = ast.parse(source)
    if typed:
        _DivisionStamper().visit(tree)
    tree = SafeDivisionTransformer().visit(tree)
    tree = FloatToleranceTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = {}
    exec(compile(tree, '<probe>', 'exec'), namespace)  # noqa: the test's own source
    return namespace


def __test_helper_emit(source: str, typed: bool = True) -> str:
    """The division pass's emission for a module source.

    :param source: The module source.
    :param typed: Whether the division operands carry a numeric type.
    :return: The unparsed emission.
    """
    tree = ast.parse(source)
    if typed:
        _DivisionStamper().visit(tree)
    return ast.unparse(SafeDivisionTransformer().visit(tree))


def __test_safe_div_matches_reference__():
    """``safe_div`` answers exactly what the reference answers, including the
    interned na, the zero-divisor signs, ``TypeError`` and overflow."""
    values = __test_helper_VALUES + __test_helper_FOREIGN
    for a in values:
        for b in values:
            assert __test_helper_key(safe_div, a, b) \
                == __test_helper_key(__test_helper_reference_div, a, b), (a, b)


def __test_safe_division_inline_matches_reference__():
    """Every inline shape answers exactly what the reference answers."""
    namespace = __test_helper_compile(
        "def ident(x):\n"
        "    return x\n"
        "def names(a, b):\n"
        "    return a / b\n"
        "def impure_left(a, b):\n"
        "    return ident(a) / b\n"
        "def impure_right(a, b):\n"
        "    return a / ident(b)\n"
        "def impure_both(a, b):\n"
        "    return ident(a) / ident(b)\n"
        "def float_literal(a, b):\n"
        "    return a / 4.0\n"
        "def int_literal(a, b):\n"
        "    return a / 3\n"
        "def literal_left(a, b):\n"
        "    return 2.0 / b\n"
        "def chained(a, b):\n"
        "    return a / b / b\n")
    shapes = {
        'names': lambda x, y: __test_helper_reference_div(x, y),
        'impure_left': lambda x, y: __test_helper_reference_div(x, y),
        'impure_right': lambda x, y: __test_helper_reference_div(x, y),
        'impure_both': lambda x, y: __test_helper_reference_div(x, y),
        'float_literal': lambda x, _: __test_helper_reference_div(x, 4.0),
        'int_literal': lambda x, _: __test_helper_reference_div(x, 3),
        'literal_left': lambda _, y: __test_helper_reference_div(2.0, y),
        'chained': lambda x, y: __test_helper_reference_div(
            __test_helper_reference_div(x, y), y),
    }
    for name, reference in shapes.items():
        emitted = namespace[name]
        for a in __test_helper_VALUES:
            for b in __test_helper_VALUES:
                assert __test_helper_key(emitted, a, b) \
                    == __test_helper_key(reference, a, b), (name, a, b)


def __test_safe_division_inline_shape__():
    """A typed division inside a function is written out; the fallback call
    receives the operands already evaluated."""
    emitted = __test_helper_emit("def f(a, b):\n    return a / b\n")
    assert '(a / (b or safe_convert.zero_divisor))' in emitted
    assert 'else safe_convert.safe_div(a, b)' in emitted
    # A literal nonzero divisor needs no zero-divisor swap
    emitted = __test_helper_emit("def f(a):\n    return a / 2\n")
    assert 'zero_divisor' not in emitted
    assert 'else safe_convert.safe_div(a, 2)' in emitted
    # An impure operand is evaluated exactly once
    emitted = __test_helper_emit("def f(g, h):\n    return g() / h()\n")
    assert emitted.count('g()') == 1 and emitted.count('h()') == 1


def __test_safe_division_inline_evaluation_order__():
    """Both operands run once, left first, on every path of the expression."""
    namespace = __test_helper_compile(
        "def probe(g, h):\n    return g() / h()\n"
        "def probe_name(a, h):\n    return a / h()\n")
    for divisor in (2.0, 0.0, -0.0, 0, float('nan'), NA(float), NA(bool)):
        order: list[str] = []

        def g() -> float:
            order.append('g')
            return 3.0

        def h(value: Any = divisor) -> Any:
            order.append('h')
            return value

        assert __test_helper_key(namespace['probe'], g, h) \
            == __test_helper_key(__test_helper_reference_div, 3.0, divisor)
        assert order == ['g', 'h'], (divisor, order)
        order.clear()
        assert __test_helper_key(namespace['probe_name'], 3.0, h) \
            == __test_helper_key(__test_helper_reference_div, 3.0, divisor)
        assert order == ['h'], (divisor, order)


def __test_safe_division_keeps_call__():
    """Every site the inline form cannot serve keeps the ``safe_div`` call."""
    call = 'safe_convert.safe_div('
    # Module level: a temporary would become a global
    assert __test_helper_emit("x = a / b\n").strip().endswith("x = safe_convert.safe_div(a, b)")
    # Class bodies, lambdas and comprehensions
    for body in ("class C:\n        y = a / b\n",
                 "f = lambda: a / b\n",
                 "y = [v / b for v in a]\n"):
        emitted = __test_helper_emit(f"def f(a, b):\n    {body}")
        assert 'zero_divisor' not in emitted and call in emitted, body
    # A default argument is evaluated in the enclosing scope
    emitted = __test_helper_emit("def f(a, b=c / d):\n    return a\n")
    assert 'zero_divisor' not in emitted and call in emitted
    # Operands the type pass did not prove numeric
    emitted = __test_helper_emit("def f(a, b):\n    return a / b\n", typed=False)
    assert 'zero_divisor' not in emitted and call in emitted
    # A literal zero divisor is always the fallback
    emitted = __test_helper_emit("def f(a):\n    return a / 0.0\n")
    assert 'zero_divisor' not in emitted and call in emitted
    # A literal over a nonzero literal stays a plain division
    emitted = __test_helper_emit("def f():\n    return 1 / 2\n")
    assert call not in emitted
    # A literal over a literal zero is not folded, so it takes the fallback too
    for divisor in ('0', '0.0'):
        emitted = __test_helper_emit(f"def f():\n    return 1 / {divisor}\n")
        assert 'zero_divisor' not in emitted and call in emitted, divisor


def __test_safe_division_literal_zero_divisor__():
    """A literal division by a literal zero yields the Pine quotient instead
    of raising, at module level and in a function alike."""
    source = ("pos = 1.0 / 0.0\nneg = -1.0 / 0.0\nnan = 0.0 / 0.0\nint_pos = 5 / 0\n"
              "def f():\n    return 1.0 / 0.0, -1.0 / 0.0, 0.0 / 0.0, 5 / 0\n")
    namespace = __test_helper_compile(source)
    assert namespace['pos'] == inf and namespace['neg'] == -inf
    assert namespace['nan'] is na_float and namespace['int_pos'] == inf
    pos, neg, nan, int_pos = namespace['f']()
    assert pos == inf and neg == -inf and nan is na_float and int_pos == inf
    # A method body is a function scope again, even inside a class
    emitted = __test_helper_emit(
        "class C:\n    def m(self, a, b):\n        return a / b\n")
    assert 'zero_divisor' in emitted


def __test_safe_division_untyped_call_matches_reference__():
    """The call form serves operands outside the numeric domain exactly."""
    namespace = __test_helper_compile("def probe(a, b):\n    return a / b\n", typed=False)
    values = __test_helper_VALUES + __test_helper_FOREIGN
    for a in values:
        for b in values:
            assert __test_helper_key(namespace['probe'], a, b) \
                == __test_helper_key(__test_helper_reference_div, a, b), (a, b)


def __test_safe_division_zero_divisor__():
    """The zero-divisor stand-in turns every numeric division into a nan the
    self-equality test rejects, and leaves an ``NA`` numerator to itself."""
    for numerator in (1.0, -1.0, 0.0, 0, 5, True, 10 ** 400, inf):
        quotient = numerator / zero_divisor
        assert not quotient == quotient, numerator
    assert isinstance(NA(None) / zero_divisor, NA)
