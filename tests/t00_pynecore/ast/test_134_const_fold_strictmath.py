"""
A constant ``math.log``/``log10``/``pow``/``tan`` call and a constant ``%`` fold the way
TradingView's parser folds them.

MEASURED on TradingView (CAPITALCOM:EURUSD 60, last bar), each value compared against an
exact literal reference below 1e-3, where the embedding is not quantized: a constant call
embeds the StrictMath (fdlibm) double, while the same call on a series argument returns the
Intel-LIBM double the runtime reproduces. The parser's ``log10`` is
``StrictMath.log(x) / StrictMath.log(10)`` (8/8 arguments where that, the product with
``1 / log(10)`` and ``StrictMath.log10`` all differ). A constant ``%`` is the truncated
remainder like a series one: ``-7 % 3`` is -1, ``7 % -3`` is 1, ``-7.5 % 2`` is -1.5.
"""
import ast
import math

import pytest

from pynecore.core import fdlibm, pine_math
from pynecore.transformers.const_fold import ConstFoldTransformer

# StrictMath values (JDK 8 x86_64 and JDK 21): the fdlibm port reproduces them bit for bit
_STRICTMATH = (
    ('log', 1.0004671006, None, 0.0004669915424740786),
    ('log', 2.5, None, 0.9162907318741551),
    ('log', 1e-310, None, -713.8013788281542),
    ('log', 7.389, None, 1.9999924078065106),
    ('log', 0.3, None, -1.2039728043259361),
    ('log', 1e+300, None, 690.7755278982137),
    ('log', 1.0000001, None, 9.999999505838704e-08),
    ('tan', 0.0005288672, None, 0.0005288672493081483),
    ('tan', 0.316603, None, 0.3276235751140051),
    ('tan', 0.7, None, 0.8422883804630794),
    ('tan', 1.365099, None, 4.79275192439207),
    ('tan', 2.914216, None, -0.23137788824011446),
    ('tan', 98139.500407, None, -0.8638823749183878),
    ('tan', 1e-09, None, 1e-09),
    ('tan', -0.9, None, -1.2601582175503392),
    ('pow', 0.0437, 2.5, 0.00039921200465363015),
    ('pow', 0.0286, 3.0, 2.3393656e-05),
    ('pow', 2.0, 0.5, 1.4142135623730951),
    ('pow', -2.0, 3.0, -8.0),
    ('pow', 1.0000001, 10000000000.0, math.inf),
    ('pow', 10.0, -3.7, 0.00019952623149688788),
    ('pow', 0.5, 1074.5, 5e-324),
    ('pow', 3.7, 2.0, 13.690000000000001),
    ('pow', 1.5, 300.25, 7.437047091715138e+52),
)

# The constant calls measured on TradingView, with the double the venue embedded
_VENUE_CONSTANTS = (
    ('log', '1.0004671006', 0.0004669915424740786),
    ('log', '1.0002423361', 0.00024230674135023667),
    ('log10', '1.0001442558', 6.26449795803409e-05),
    ('log10', '1.0005712571', 0.0002480229705795736),
    ('log10', '1.0005183719', 0.0002250677263897057),
    ('log10', '1.0001336901', 5.805699197901573e-05),
    ('log10', '1.0005521505', 0.00023972973797159314),
    ('log10', '1.0000692895', 3.0091005020557846e-05),
    ('log10', '1.0002681896', 0.00011645764771682663),
    ('log10', '1.0002702578', 0.00011735561381051342),
    ('log10', '1.000139718', 6.067451786011606e-05),
    ('log10', '1.0004904254', 0.00021293683445595184),
    ('log10', '1.0005474024', 0.0002376687973987255),
    ('tan', '0.0005288672', 0.0005288672493081483),
    ('tan', '0.0005054318', 0.0005054318430394268),
    ('tan', '0.0004518089', 0.0004518089307427792),
    ('pow', '0.0437, 2.5', 0.00039921200465363015),
    ('pow', '0.019, 3.5', 9.454479039058682e-07),
    ('pow', '0.0286, 3.0', 2.3393656e-05),
)


def __test_helper_fold(source: str) -> ast.Module:
    return ConstFoldTransformer().visit(ast.parse(source))


def __test_helper_folded_value(expression: str) -> object:
    """The constant the fold embeds for ``X = expression``, or the node it left."""
    tree = __test_helper_fold(f'from pynecore import lib\nX = {expression}\n')
    value = tree.body[1].value
    return value.value if isinstance(value, ast.Constant) else value


@pytest.mark.parametrize('name,x,y,expected', _STRICTMATH)
def __test_fdlibm_port_is_strictmath__(name, x, y, expected):
    result = getattr(fdlibm, name)(x, y) if y is not None else getattr(fdlibm, name)(x)
    assert result == expected


@pytest.mark.parametrize('name,arguments,expected', _VENUE_CONSTANTS)
def __test_constant_call_embeds_the_venue_double__(name, arguments, expected):
    assert __test_helper_folded_value(f'lib.math.{name}({arguments})') == expected


def __test_series_and_constant_calls_differ_where_measured__():
    """the runtime functions are not the fold's: a series call of the same argument
    returns the Intel-LIBM double"""
    assert pine_math.log(1.0004671006) != 0.0004669915424740786
    assert pine_math.log10(1.0001442558) != 6.26449795803409e-05
    assert pine_math.pow(0.0437, 2.5) != 0.00039921200465363015


@pytest.mark.parametrize('expression,expected', [
    ('-7 % 3', -1), ('7 % -3', 1), ('7 % 3', 1), ('-7.5 % 2', -1.5), ('7 % 2.5', 2.0),
    ('-1e-300 % 1', -1e-300),
])
def __test_constant_remainder_is_truncated__(expression, expected):
    value = __test_helper_folded_value(expression)
    assert value == expected and type(value) is type(expected)


def __test_constant_zero_remainder_is_unsigned__():
    value = __test_helper_folded_value('-6.0 % 3')
    assert value == 0.0 and math.copysign(1.0, value) == 1.0


def __test_constant_remainder_by_zero_is_not_folded__():
    assert isinstance(__test_helper_folded_value('7 % 0'), ast.BinOp)
