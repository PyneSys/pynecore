"""
A constant ``math.log``/``log10``/``pow``/``tan`` call and a constant ``%`` fold the way
TradingView's parser folds them.

MEASURED on TradingView (CAPITALCOM:EURUSD 60, last bar), each value compared against an
exact literal reference below 1e-3, where the embedding is not quantized: a constant call
embeds the StrictMath (fdlibm) double, while the same call on a series argument returns the
Intel-LIBM double the runtime reproduces. The parser's ``pow`` is StrictMath's without the
tail of the reduced exponent, one ulp off it where that tail moves the last rounding
(``pow(10, -4)`` is 9.999999999999999e-05, StrictMath's 1e-4). The parser's ``log10`` is
``StrictMath.log(x) / StrictMath.log(10)`` (8/8 arguments where that, the product with
``1 / log(10)`` and ``StrictMath.log10`` all differ). A constant ``%`` is the truncated
remainder like a series one: ``-7 % 3`` is -1, ``7 % -3`` is 1, ``-7.5 % 2`` is -1.5.
Constant int arithmetic is double arithmetic, as on a series int (probes int_a / int_b,
2026-10-26): ``9007199254740993 - 9007199254740992`` is 0. A constant ``math.round``
without a precision rounds the number to 15 decimal places before it rounds it to an
integer, so ``3.4999999999999996`` folds to 4 where a series rounds it to 3. With a
positive precision it rounds the scaled number twice, to the precision's places and then
to an integer, so ``math.round(1.245, 1)`` folds to 1.3 where a series answers 1.2.
"""
import ast
import math

import pytest

from pynecore.core import fdlibm, pine_math
from pynecore.lib import math as lib_math
from pynecore.transformers.const_fold import ConstFoldTransformer, quantize_embed
from pynecore.transformers.pine_type_rules import BOOL, INT, get_ty

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

# The constant pow calls measured on TradingView where the venue's double is not
# StrictMath's (probes roundconst7/8 and pw1-pw7, CAPITALCOM:EURUSD 60, 2026-10-31). Equal
# products y * log2(x) embed the same double: pow(10, -4), pow(1e8, -0.5), pow(100, -2) and
# pow(1e16, -0.25). Negative bases, the |y| > 2**31 path of bases next to one, subnormal
# and huge results drop the tail all the same.
_PARSER_POW = (
    (10.0, 3.5, 3162.277660168379),
    (10.0, -4.0, 9.999999999999999e-05),
    (10.0, 7.7, 50118723.362727255),
    (10.0, 3.75, 5623.41325190349),
    (10.0, -6.9492, 1.1240871934697294e-07),
    (10.0, -12.11, 7.762471166286928e-13),
    (10.0, -5.3012, 4.9980431357823956e-06),
    (10.0, -5.598, 2.5234807724805758e-06),
    (100000000.0, -0.5, 9.999999999999999e-05),
    (10.0, -11.0, 1e-11),
    (10.0, -23.0, 1.0000000000000001e-23),
    (5.0, -17.8107, 3.5551255601153003e-13),
    (6.0, -13.685, 2.2438826393740117e-11),
    (2.5, -10.89, 4.6390954209948444e-05),
    (20.0, -6.3173, 6.039530911209967e-09),
    (20.0, -10.652, 1.3849321150303411e-14),
    (20.0, -2.7513, 0.000263315347475703),
    (7.0, -15.52, 7.657348603941436e-14),
    (1.1, -315.6631, 8.587300993670791e-14),
    (0.07, 10.005, 2.7874423261898983e-12),
    (1.7, -50.5556, 2.2362302108160535e-12),
    (2.0, -23.1334, 1.0868076789460136e-07),
    (2.0, -32.7482, 1.3861476594977307e-10),
    (2.0, -20.8738, 5.204273075281934e-07),
    (2.0, -11.757, 0.00028892847210756136),
    (100.0, -2.0, 9.999999999999999e-05),
    (1e+16, -0.25, 9.999999999999999e-05),
    (100.0, 1.75, 3162.277660168379),
    (100000000000000.0, 0.25, 3162.277660168379),
    (10000000000.0, 0.77, 50118723.362727255),
    (-36.0, -15.0, -4.5233739070769964e-24),
    (-54.3, 11.0, -1.2100355380322263e+19),
    (-48.54, -4.0, 1.8013616475160188e-07),
    (-40.0, -11.0, -2.384185791015625e-18),
    (0.999999999562533, -70588518506.0, 25767729031418.562),
    (1.0000000005402399, -188973603330.0, 4.596327229381187e-45),
    (1.000000000435648, -262015666165.0, 2.6715629043993666e-50),
    (1.0000000001055014, -292784380502.0, 3.845999570327458e-14),
    (351.2, 11.644, 4.369549687753981e+29),
    (35.48, 161.293, 1.0034750868428818e+250),
    (72.2, -16.0, 1.834023258213995e-30),
    (159.65, -32.7054, 8.799927817062741e-73),
    (44.95, -6.2693, 4.350525478732898e-11),
    (203.855, -19.04, 1.0728059017174899e-44),
)


def __test_helper_fold(source: str) -> ast.Module:
    return ConstFoldTransformer().visit(ast.parse(source))


def __test_helper_folded_node(expression: str) -> ast.expr:
    """The right side the fold leaves for ``X = expression``."""
    tree = __test_helper_fold(f'from pynecore import lib\nX = {expression}\n')
    return tree.body[1].value


def __test_helper_folded_value(expression: str) -> object:
    """The constant the fold embeds for ``X = expression``, or the node it left."""
    value = __test_helper_folded_node(expression)
    return value.value if isinstance(value, ast.Constant) else value


@pytest.mark.parametrize('name,x,y,expected', _STRICTMATH)
def __test_fdlibm_port_is_strictmath__(name, x, y, expected):
    result = getattr(fdlibm, name)(x, y) if y is not None else getattr(fdlibm, name)(x)
    assert result == expected


@pytest.mark.parametrize('name,arguments,expected', _VENUE_CONSTANTS)
def __test_constant_call_embeds_the_venue_double__(name, arguments, expected):
    assert __test_helper_folded_value(f'lib.math.{name}({arguments})') == expected


@pytest.mark.parametrize('x,y,expected', _PARSER_POW)
def __test_constant_pow_drops_the_tail_of_the_reduced_exponent__(x, y, expected):
    """a constant ``math.pow`` embeds the parser's double where StrictMath's differs"""
    assert __test_helper_folded_value(f'lib.math.pow({x!r}, {y!r})') == expected
    assert quantize_embed(fdlibm.pow(x, y)) != expected


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


# The constant int expressions measured on TradingView, with the value the venue plotted
@pytest.mark.parametrize('expression,expected', [
    ('100000000000 * 100000000000 + 1 - 100000000000 * 100000000000', 0),
    ('9007199254740993 - 9007199254740992', 0),
    ('4611686018427387904 * 4 / 4611686018427387904', 4.0),
    ('4611686018427387904 * 4 - 4611686018427387904 * 4 + 7', 7),
    ('10000000000000000000000 % 7', 4),
])
def __test_constant_int_arithmetic_is_double__(expression, expected):
    node = __test_helper_folded_node(expression)
    assert isinstance(node, ast.Constant)
    assert node.value == expected and type(node.value) is type(expected)
    assert get_ty(node) == INT


@pytest.mark.parametrize('literal,expected', [
    ('9007199254740993', 9007199254740992),
    ('-9007199254740995', -9007199254740996),
    ('18446744073709551615', 18446744073709551616),
    ('100000000000000000000000000000', 99999999999999991433150857216),
    ('9007199254740991', 9007199254740991),
])
def __test_int_literal_embeds_its_double__(literal, expected):
    node = __test_helper_folded_node(literal)
    assert isinstance(node, ast.Constant)
    assert node.value == expected and type(node.value) is int
    assert get_ty(node) == INT


def __test_int_beyond_the_double_range_is_not_folded__():
    """an int literal or int result that rounds to an infinity stays in the code"""
    literal = '1' + '0' * 400
    assert __test_helper_folded_value(literal) == int(literal)
    assert isinstance(__test_helper_folded_value(f'{literal} - {literal}'), ast.BinOp)
    huge = str(10 ** 200)
    assert isinstance(__test_helper_folded_value(f'{huge} * {huge}'), ast.BinOp)


@pytest.mark.parametrize('expression,expected', [
    ('lib.math.round(0.49999999999999994)', 1.0), ('lib.math.round(-0.49999999999999994)', -1.0),
    ('lib.math.round(0.5 - lib.math.pow(2, -53))', 1.0), ('lib.math.round(0.4999999999999995)', 1.0),
    ('lib.math.round(0.4999999999999994)', 0.0), ('lib.math.round(0.49999999999999944)', 0.0),
    ('lib.math.round(3.4999999999999996)', 4.0), ('lib.math.round(-3.4999999999999996)', -4.0),
    ('lib.math.round(2.4999999999999996, 0)', 3.0), ('lib.math.round(2.499999999999999)', 2.0),
    ('lib.math.round(1.4999999999999996)', 2.0), ('lib.math.round(1.4999999999999993)', 1.0),
    ('lib.math.round(1000.4999999999999)', 1000.0), ('lib.math.round(452523.49999999994)', 452523.0),
    ('lib.math.round(0.5 - 1e-11)', 0.0),
])
def __test_constant_integer_round_rounds_to_15_places_first__(expression, expected):
    """a constant ``math.round`` with no precision, or precision 0, rounds the number to
    15 decimal places, half up, before it rounds it to an integer; a series one compares
    the exact fraction (probes roundsplit, roundconst, roundconst2, CAPITALCOM:EURUSD 60)"""
    assert __test_helper_folded_value(expression) == expected


def __test_series_integer_round_compares_the_exact_fraction__():
    assert lib_math.round(0.49999999999999994) == 0.0
    assert lib_math.round(3.4999999999999996) == 3.0
    assert lib_math.round(0.4999999999999995, 0) == 0.0


# The constants TradingView plotted (probes roundconst-roundconst9, CAPITALCOM:EURUSD 60)
@pytest.mark.parametrize('number,precision,expected', [
    ('1.245', '1', 1.3), ('1.2451', '1', 1.3), ('1.24499', '1', 1.2), ('-1.2451', '1', -1.3),
    ('0.245', '1', 0.3), ('0.145', '1', 0.1), ('1.23495', '2', 1.24), ('-1.23495', '2', -1.24),
    ('1.2349499', '2', 1.23), ('0.08495', '2', 0.08), ('1.2344995', '3', 1.235),
    ('1.23444999501', '4', 1.2345), ('1.23444499995', '5', 1.23445), ('63.986275', '5', 63.98628),
    ('0.12345499999999', '5', 0.12346), ('123456.37453499998', '5', 123456.37454),
    ('4503599.960634999', '5', 4503599.96063), ('4503599.225499999', '3', 4503599.226),
    ('3.4140024999999996', '6', 3.414003), ('63.134680499999995', '6', 63.134681),
    ('-63.134680499999995', '6', -63.134681), ('0.529078455', '8', 0.52907845),
    ('-0.529078455', '8', -0.52907845), ('0.333323265', '8', 0.33332327),
    ('0.517401755', '8', 0.51740176), ('0.945807305', '8', 0.94580731),
    ('0.6435383349999999', '8', 0.64353834), ('0.6519647549999997', '8', 0.65196475),
    ('0.3024663349999995', '8', 0.30246633), ('0.7809848565499995', '10', 0.7809848565),
    ('0.12345678901234568', '17', 0.123456789012346), ('0.12345678901234568', '100', 0.123456789012346),
    ('7', '2', 7.0), ('2.34567', '38 / 1000', 2.748661470365999),
    ('123.456789', '38 / 1000', 123.68976616646995), ('2.34567', '361 / 1000', 2.177559368427843),
    ('4.006', '6 / 5', 4.038127004673237), ('6.3712', '3 / 2', 6.387800873540127),
    ('4.211', '27 / 10', 4.211998746899304), ('2.4076', '7 / 2', 2.4077582104522044),
    ('7.367', '7 / 2', 7.367158264894274),
])
def __test_constant_round_to_a_precision_rounds_twice__(number, precision, expected):
    """a constant ``math.round(number, precision)`` rounds the number to 15 decimal places,
    scales it by ``10**precision``, rounds the product half up to ``int(precision)`` places
    and then to an integer: the first rounding moves the tie"""
    assert __test_helper_folded_value(f'lib.math.round({number}, {precision})') == expected


def __test_series_round_to_a_precision_rounds_once__():
    assert lib_math.round(1.245, 1) == 1.2
    assert lib_math.round(1.23495, 2) == 1.23
    assert lib_math.round(63.986275, 5) == 63.98627
    assert lib_math.round(2.34567, 0.038) == 2.0


@pytest.mark.parametrize('expression,expected', [
    ('cast_bool(1e-10)', True), ('cast_bool(-1e-10)', True), ('cast_bool(2.5e-17)', True),
    ('cast_bool(0.1 + 0.2 - 0.3)', True), ('cast_bool(lib.math.abs(1e-10))', True),
    ('cast_bool(0.0)', False), ('cast_bool(-0.0)', False), ('cast_bool(0)', False),
])
def __test_constant_bool_cast_is_exact__(expression, expected):
    """a constant ``bool()`` is exact, any non-zero value is true (MEASURED v5 and v6,
    CAPITALCOM:EURUSD 60: literal, constant variable, expression and ``math.*`` call)"""
    tree = __test_helper_fold('from pynecore import lib\nfrom pynecore.core.pine_cast import cast_bool\n'
                              f'X = {expression}\n')
    value = tree.body[2].value
    assert isinstance(value, ast.Constant) and value.value is expected
    assert get_ty(value) == BOOL


def __test_series_bool_cast_is_not_folded__():
    """a series ``bool()`` stays the runtime cast, which is tolerant (``close * 0 + 1e-10``
    is false on TradingView), and a name that is not the ``pine_cast`` import is no cast"""
    tree = __test_helper_fold('from pynecore.lib import close\n'
                              'from pynecore.core.pine_cast import cast_bool\n'
                              'X = cast_bool(close * 0 + 1e-10)\n'
                              'def cast_float(v):\n    return v\n'
                              'Y = cast_float(1e-10)\n')
    assert isinstance(tree.body[2].value, ast.Call)
    assert isinstance(tree.body[4].value, ast.Call)
    shadowed = __test_helper_fold('from pynecore.core.pine_cast import cast_bool\n'
                                  'cast_bool = bool\nX = cast_bool(1e-10)\n')
    assert isinstance(shadowed.body[2].value, ast.Call)


def __test_class_local_bool_cast_shadowing__():
    """a class-local binding shadows the cast in the class body, while a method keeps
    resolving it in the module (a method does not see the class namespace)"""
    tree = __test_helper_fold('from pynecore.core.pine_cast import cast_bool\n'
                              'class K:\n'
                              '    cast_bool = lambda x: False\n'
                              '    flag = cast_bool(1.0)\n'
                              '    def m(self):\n'
                              '        return cast_bool(1.0)\n')
    klass = tree.body[1]
    assert isinstance(klass, ast.ClassDef)
    flag, method = klass.body[1], klass.body[2]
    assert isinstance(flag, ast.Assign) and isinstance(flag.value, ast.Call)
    assert isinstance(method, ast.FunctionDef)
    statement = method.body[0]
    assert isinstance(statement, ast.Return)
    returned = statement.value
    assert isinstance(returned, ast.Constant) and returned.value is True


def __test_class_local_bool_cast_shadowing_in_lambda_and_comprehension__():
    """a lambda or comprehension in a class body resolves the cast in the module, as a
    method does; only the outermost iterable of a comprehension sees the class binding"""
    tree = __test_helper_fold('from pynecore.core.pine_cast import cast_bool\n'
                              'class K:\n'
                              '    cast_bool = lambda x: False\n'
                              '    anonymous = staticmethod(lambda: cast_bool(1e-10))\n'
                              '    listed = [cast_bool(1e-10) for _ in (1,)]\n'
                              '    outer = [x for x in (cast_bool(1e-10),)]\n')
    klass = tree.body[1]
    assert isinstance(klass, ast.ClassDef)
    anonymous, listed, outer = klass.body[1], klass.body[2], klass.body[3]
    assert isinstance(anonymous, ast.Assign) and isinstance(anonymous.value, ast.Call)
    lambda_node = anonymous.value.args[0]
    assert isinstance(lambda_node, ast.Lambda)
    assert isinstance(lambda_node.body, ast.Constant) and lambda_node.body.value is True
    assert isinstance(listed, ast.Assign) and isinstance(listed.value, ast.ListComp)
    assert isinstance(listed.value.elt, ast.Constant) and listed.value.elt.value is True
    assert isinstance(outer, ast.Assign) and isinstance(outer.value, ast.ListComp)
    iterable = outer.value.generators[0].iter
    assert isinstance(iterable, ast.Tuple) and isinstance(iterable.elts[0], ast.Call)


def __test_class_local_constant_shadowing_in_lambda_and_comprehension__():
    """a lambda or comprehension in a class body reads the module constant, not the
    class binding of the same name; only the outermost iterable sees the class one"""
    tree = __test_helper_fold('from pynecore.core.pine_cast import cast_bool\n'
                              'X = 1\n'
                              'E = 1e-10\n'
                              'class K:\n'
                              '    X = 0\n'
                              '    E = 0\n'
                              '    anonymous = staticmethod(lambda: X)\n'
                              '    listed = [X for _ in (0,)]\n'
                              '    outer = [x for x in (X,)]\n'
                              '    cast = staticmethod(lambda: cast_bool(E))\n')
    klass = tree.body[3]
    assert isinstance(klass, ast.ClassDef)
    anonymous, listed, outer, cast = klass.body[2:6]
    assert isinstance(anonymous, ast.Assign) and isinstance(anonymous.value, ast.Call)
    lambda_node = anonymous.value.args[0]
    assert isinstance(lambda_node, ast.Lambda)
    assert isinstance(lambda_node.body, ast.Constant) and lambda_node.body.value == 1
    assert isinstance(listed, ast.Assign) and isinstance(listed.value, ast.ListComp)
    assert isinstance(listed.value.elt, ast.Constant) and listed.value.elt.value == 1
    assert isinstance(outer, ast.Assign) and isinstance(outer.value, ast.ListComp)
    iterable = outer.value.generators[0].iter
    assert isinstance(iterable, ast.Tuple)
    first = iterable.elts[0]
    assert isinstance(first, ast.Constant) and first.value == 0
    assert isinstance(cast, ast.Assign) and isinstance(cast.value, ast.Call)
    cast_lambda = cast.value.args[0]
    assert isinstance(cast_lambda, ast.Lambda)
    assert isinstance(cast_lambda.body, ast.Constant) and cast_lambda.body.value is True


def __test_deferred_body_folds_only_names_never_rebound__():
    """a lambda or generator body runs later and reads the value a name has then: it
    folds a name only when its scope never rebinds it; a list comprehension and the
    outermost iterable of a generator run at once and fold the current value"""
    tree = __test_helper_fold('X = 1.5\nS = 2.0\n'
                              'f = lambda: X\n'
                              'g = (X for _ in range(int(X)))\n'
                              'h = [X for _ in (0,)]\n'
                              'k = lambda: S\n'
                              'X = 2.5\n'
                              'def m():\n'
                              '    Y = 1.5\n'
                              '    T = 3.0\n'
                              '    f = lambda: Y\n'
                              '    k = lambda: T + S\n'
                              '    Y = 2.5\n'
                              '    return f, k\n')
    f, g, h, k = (statement.value for statement in tree.body[2:6])  # type: ignore[attr-defined]
    assert isinstance(f, ast.Lambda) and isinstance(f.body, ast.Name)
    assert isinstance(g, ast.GeneratorExp) and isinstance(g.elt, ast.Name)
    call = g.generators[0].iter
    assert isinstance(call, ast.Call)
    inner = call.args[0]
    assert isinstance(inner, ast.Call)
    argument = inner.args[0]
    assert isinstance(argument, ast.Constant) and argument.value == 1.5
    assert isinstance(h, ast.ListComp)
    assert isinstance(h.elt, ast.Constant) and h.elt.value == 1.5
    assert isinstance(k, ast.Lambda)
    assert isinstance(k.body, ast.Constant) and k.body.value == 2.0
    method = tree.body[7]
    assert isinstance(method, ast.FunctionDef)
    local_f, local_k = (statement.value for statement in method.body[2:4])  # type: ignore[attr-defined]
    assert isinstance(local_f, ast.Lambda) and isinstance(local_f.body, ast.Name)
    assert isinstance(local_k, ast.Lambda)
    assert isinstance(local_k.body, ast.Constant) and local_k.body.value == 5.0
