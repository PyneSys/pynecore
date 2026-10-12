"""A constant folded to zero is embedded as an unsigned ``0.0``; runtime arithmetic keeps -0.0."""
import ast
import math

import pytest

from pynecore.transformers.const_fold import ConstFoldTransformer, quantize_embed


def _fold(source: str) -> ast.Module:
    return ConstFoldTransformer().visit(ast.parse(source))


def _execute(tree: ast.Module) -> dict:
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), '<constant folding>', 'exec'), namespace)
    return namespace


def _is_positive_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) == 1.0


def __test_embedding_drops_the_zero_sign__():
    assert _is_positive_zero(quantize_embed(-0.0))
    assert _is_positive_zero(quantize_embed(0.0))
    assert quantize_embed(0) == 0 and type(quantize_embed(0)) is int


@pytest.mark.parametrize('expression', [
    '-0.0', '0.0 * -1', '-1 * 0.0', '-(3.0 * 0)', '0.0 / -5', '-0.0 - 0.0', '-0.0 * 5',
    '-0.0 / 1', 'lib.math.ceil(-0.5)',
])
def __test_constant_expression_embeds_an_unsigned_zero__(expression):
    """the same forms TradingView embeds as +0.0"""
    namespace = _execute(_fold(f'from pynecore import lib\nX = {expression}\n'))
    assert _is_positive_zero(float(namespace['X']))


def __test_constant_chain_and_var_initializer_embed_an_unsigned_zero__():
    """``-c0`` of a constant ``c0 = 0.0`` and a ``var``-style annotated initializer"""
    tree = _fold('''C0 = 0.0
def main():
    neg = -C0
    v: float = -0.0
    return neg, v
''')
    neg, v = _execute(tree)['main']()
    assert _is_positive_zero(neg) and _is_positive_zero(v)


def __test_constant_argument_of_a_runtime_call_embeds_an_unsigned_zero__():
    """an ``input.float(-0.0)`` default is a constant argument inside a runtime call"""
    tree = _fold('def main(f):\n    return f(-0.0)\n')
    assert _is_positive_zero(_execute(tree)['main'](lambda x: x))


def __test_an_already_signed_literal_is_replaced__():
    """a ``Constant(-0.0)`` left by an earlier pass is not mistaken for the embedded ``0.0``"""
    tree = ast.parse('X = 0.0\n')
    tree.body[0].value = ast.Constant(value=-0.0)  # type: ignore[attr-defined]
    assert _is_positive_zero(_execute(ConstFoldTransformer().visit(tree))['X'])


def __test_runtime_arithmetic_keeps_the_ieee_sign__():
    """an operand that is not a constant leaves the negation to the runtime"""
    tree = _fold('def main(close):\n    return -(close * 0.0)\n')
    result = _execute(tree)['main'](1.12)
    assert result == 0.0 and math.copysign(1.0, result) == -1.0
