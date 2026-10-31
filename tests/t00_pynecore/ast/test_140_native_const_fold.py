"""
The native ``ConstFold`` step (``rust/crates/pyne-transform/src/steps/const_fold.rs``)
against ``transformers/const_fold.py``.

The evaluation is held against ``_try_eval`` + ``quantize_embed`` one expression at a
time over seeded random ``lib.math`` calls and operator chains with Pine's int (a double)
and float semantics at their edges; the step over modules that reach every statement
kind, scope rule and propagation path, compared by ``native.describe``.
"""
import ast
import math
import random
import struct
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pynecore.transformers import const_fold, native
from pynecore.transformers.const_fold import ConstFoldTransformer, quantize_embed

from .test_137_native_transform import __test_helper_require_native

#: Every ``lib.math`` name the fold knows, a few it leaves alone, and a non-function
__test_helper_FUNCTIONS = (
    'sin', 'cos', 'tan', 'exp', 'log', 'log10', 'pow', 'asin', 'acos', 'atan',
    'sqrt', 'abs', 'floor', 'ceil', 'min', 'max', 'avg', 'sign', 'round', 'todegrees',
    'toradians', 'random', 'sum', 'round_to_mintick', 'pi',
)

__test_helper_OPERATORS = ('+', '-', '*', '/', '%', '//', '**', '&')

#: A module reaching every branch of the step
__test_helper_MODULE = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, close, bar_index
from pynecore.types import Persistent, IBPersistent, Series

A = 14 / 8
B = lib.math.sqrt(2) * 3
C: float = lib.math.pow(1.1, 3.5)
C2: int = 7
D = lib.math.round(2.5) + lib.math.floor(-0.5) + lib.math.ceil(-0.5) + lib.math.round(-2.5)
E = lib.math.max(1, 2.0, 2) - lib.math.min(3, 1.0, 1)
E2 = lib.math.max(2, 2.0)
E3 = lib.math.min(-0.0, 0.0)
F = lib.math.avg(1, 2.5, 1e-17) + lib.math.avg(0.1, 0.2) + lib.math.avg(3)
F2 = lib.math.avg(7.65, 3.0, 23.7)
F3 = lib.math.avg(1.0, 25.85, 81.95) * 3
G = -7 % 3 + 7.5 % -2 + (-6.0 % 3)
H = lib.math.todegrees(lib.math.pi / 3) + lib.math.toradians(45)
I = lib.math.log10(1000) + lib.math.log(10) + lib.math.exp(0.1)
J = (lib.math.sin(1e22) + lib.math.cos(7) + lib.math.tan(1.5) + lib.math.asin(0.3)
     + lib.math.acos(-0.7) + lib.math.atan(3))
K = lib.math.sign(-3) + lib.math.abs(-4) + lib.math.abs(-2.5) + lib.math.sign(0)
L = 0.21402930929524275
M = 1 / 3
N = lib.math.phi + lib.math.rphi + lib.math.e
O = 1e-5 * 3
P = lib.math.sqrt(-1)
Q = 1 / 0
Q2 = 1 % 0.0
R = lib.math.round(1234.5678, -2)
R2 = lib.math.round(4503599627370497)
R3 = lib.math.round(-4503599627370496.0)
S = 2 ** 3
T, U = 1, 2
V = lib.math.random()
W = -0.0
X = 0.0 * -1
Y = lib.math.max(3)
Y2 = lib.math.abs(-3, 4)
Y3 = lib.math.sin(1, 2)
Y4 = lib.math.sqrt(x=4)
Y5 = lib.math.max(*[1, 2])
Y6 = lib.math.log(0)
Y7 = lib.math.exp(1000)
Y8 = 1e308 * 10
Y9 = lib.math.floor(1e300) + lib.math.ceil(7)
Y10 = lib.math.asin(2)
Y11 = lib.math.pow(-8, 1 / 3)
Y12 = 9007199254740992 / 3 + 2 ** 3 * 4
Y13 = 2 ** 0.5 + B
Y14 = lib.math.round(3.7, 0)
AA = A + B * C - M
AB = AA / 7 + D
AC = True
AD = 'text'
AE = not 1
AF = ~5
AG = 1e309
AH = lib.math.max(1, 2) + lib.math.min(1.0, 2)
AI = lib.math.round(2.5) * 2 + lib.math.round(2.5, 0)
AJ = 1 if bar_index else 2.5
xs = [1, 2, 3]
xs[0] = A + 1
count: Persistent[int] = 0
total: Persistent[float] = 0.5
fixed: Persistent[float] = 0.25 * 2
mutated: IBPersistent[float] = 1.5
mutated = mutated + 1
series_value: Series[float] = 1.0 + 1
plain: float
plain2: float = close
REBOUND = 1
REBOUND = 2
GLOBAL_WRITTEN = 3
anno: lib.math = 4


def defaults(a=A * 2, *, b=lib.math.sqrt(9), c=None, d=close):
    return a + b + AA + REBOUND + GLOBAL_WRITTEN


def shadows(A, x=B):
    local = A + B
    B2 = local * 2
    return B2 + A


def locals_only():
    k = 10
    k2 = k * 3
    if k2 > 5:
        k3 = k2 + 1
        q = k3 / 2
    else:
        q = 1
    m = q + k2
    for i in range(3):
        k = k + i
        z = k * 2
    w = k + 1
    while w < 100:
        w = w * 2
        v = w + k2
    with open('x') as handle, open('y'):
        h = k2 + 1
    try:
        t = k2 * 2
    except ValueError as error:
        t = 0
    else:
        t2 = t + 1
    finally:
        t3 = k2 - 1
    del t
    n = k2 + 1
    n += 2
    n -= k2
    assert k2 > 0, k2 * 2
    if k2 < 0:
        raise ValueError(k2 * 3) from None
    return n + k2


def nested():
    c = 5

    def inner(d=c * 2):
        e = c + 1
        return e + AA

    def inner2():
        nonlocal c
        c = 7
        return c * 2

    return inner() + inner2() + c * 3


def writes():
    global GLOBAL_WRITTEN
    GLOBAL_WRITTEN = 4
    return GLOBAL_WRITTEN * 2


def walrus():
    p = 3
    if (p := p + 1) > 2:
        r = p * 2
    return p + r


def comprehensions():
    k = 4
    a = [k * 2 for k in range(3)]
    b = {k: k + 1 for k in range(2)}
    c = {k + A for _ in range(2)}
    d = list(x * k for x in range(2))
    f = lambda k, *rest, z=k * 2, **kw: k + A + z
    g = lambda: k * 3 + A
    return a, b, c, d, f, g


def subscripts(series_x):
    y = 3
    a = series_x[1 + 1]
    b = series_x[y]
    c = [1, 2, 3][y - 2]
    d = (series_x + 1)[y * 0]
    return a + b + c + d


def matches(x):
    m = 2
    match x:
        case [a, *rest]:
            m = a
        case {'k': v, **others}:
            m = v
        case _:
            pass
    return m * 2


def imports():
    q = 2
    import math as pymath
    from os import path
    return q * 3


def calls():
    k = 2
    return print(k * 2, sep=str(k + 1)), f'{k * 3}', {**{}}, (*[k + 1],)


def persistent_scope():
    acc: Persistent[float] = 0.0
    acc = acc + 1
    fixed2: Persistent[int] = 3
    return acc + fixed2 * 2


async def asynchronous():
    k = 1
    async with ctx() as c:
        k2 = k + 1
    async for item in source():
        k3 = k + 2
    return k * 5


class Holder:
    VALUE = 3 * 2
    OTHER = VALUE + 1

    def method(self, x=A + 1):
        return self.VALUE + A * 2


@script.indicator('fold')
def main():
    s = close * A + lib.math.sqrt(2) * B
    loop_total = 0
    for i in range(int(A * 4)):
        loop_total = loop_total + i * D
    return s + loop_total + defaults() + shadows(1) + locals_only() + nested()
'''

#: Modules where the fold must keep no module name: a wildcard import
__test_helper_WILDCARD = '''
from pynecore import lib
from pynecore.lib.math import *
A = 2 * 3


def f():
    return A + 1
'''


def __test_helper_native():
    from pynecore.transformers import _native

    return _native


def __test_helper_expression(rnd: random.Random, depth: int) -> str:
    """A random expression of literals, ``lib.math`` calls and operators."""
    pick = rnd.random()
    if depth <= 0 or pick < 0.3:
        return __test_helper_number(rnd)
    if pick < 0.65:
        name = rnd.choice(__test_helper_FUNCTIONS)
        if name == 'pi':
            return f'lib.math.{rnd.choice(("pi", "e", "phi", "rphi", "tau"))}'
        count = rnd.choice((0, 1, 1, 1, 2, 2, 3))
        args = ', '.join(__test_helper_expression(rnd, depth - 1) for _ in range(count))
        return f'lib.math.{name}({args})'
    if pick < 0.75:
        return f'{rnd.choice("-+")}({__test_helper_expression(rnd, depth - 1)})'
    op = rnd.choice(__test_helper_OPERATORS)
    return (f'({__test_helper_expression(rnd, depth - 1)} {op} '
            f'{__test_helper_expression(rnd, depth - 1)})')


def __test_helper_number(rnd: random.Random) -> str:
    """A literal at one of Python's int / float edges."""
    pick = rnd.random()
    if pick < 0.25:
        return str(rnd.randint(-10, 10))
    if pick < 0.35:
        return str(rnd.choice((2 ** 53, 2 ** 53 + 1, 2 ** 62, 2 ** 63 - 1, 2 ** 63, 2 ** 64,
                               4503599627370496, 4503599627370497, 3037000499, 3037000500)))
    if pick < 0.45:
        return repr(rnd.uniform(-10.0, 10.0))
    if pick < 0.55:
        # Prices as a script spells them: a few decimals
        return repr(round(rnd.uniform(0.0, 100.0), rnd.randint(0, 3)))
    if pick < 0.65:
        return repr(rnd.choice((0.5, 1.5, 2.5, -0.0, 0.0, 0.49999999999999994, 1e300, 1e-320,
                                1e308, 0.1, 0.21402930929524275, 4503599627370495.5)))
    if pick < 0.8:
        return repr(abs(math.ldexp(rnd.random(), rnd.randint(-60, 60))))
    value = struct.unpack('<d', struct.pack('<Q', rnd.getrandbits(64)))[0]
    return repr(abs(value)) if math.isfinite(value) else '1.0'


def __test_helper_python_eval(node: ast.expr):
    """``quantize_embed(_try_eval(node, {}))``, None for a bail; raises what it raises."""
    value = const_fold._try_eval(node, {})
    if value is const_fold._BAIL:
        return None
    return quantize_embed(value)


def __test_helper_same(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack('<d', a) == struct.pack('<d', b)
    return a == b


def __test_native_const_eval_matches_python__():
    """The native evaluation of a constant expression is Python's, value and type"""
    __test_helper_require_native()
    _native = __test_helper_native()
    rnd = random.Random(20261027)
    checked = unsupported = 0
    for _ in range(20_000):
        source = __test_helper_expression(rnd, rnd.randint(1, 4))
        node = ast.parse(source, mode='eval').body
        try:
            want = __test_helper_python_eval(node)
        except (OverflowError, ValueError, ZeroDivisionError, TypeError, MemoryError):
            # Python's fold itself raises: the native one must not fold it either
            try:
                assert _native._const_eval(node) is None, source
            except _native.Unsupported:
                pass
            continue
        try:
            got = _native._const_eval(node)
        except _native.Unsupported:
            unsupported += 1
            continue
        checked += 1
        assert __test_helper_same(got, want), f'{source}: native {got!r}, Python {want!r}'
    assert checked > 15_000 and unsupported < 2_000, (checked, unsupported)


#: Ints beyond 2**53, 2**63 and the double range: the TradingView rows of
#: ``test_134_const_fold_strictmath`` first
__test_helper_WIDE_INTS = [
    '100000000000 * 100000000000 + 1 - 100000000000 * 100000000000',
    '9007199254740993 - 9007199254740992', '4611686018427387904 * 4 / 4611686018427387904',
    '4611686018427387904 * 4 - 4611686018427387904 * 4 + 7', '10000000000000000000000 % 7',
    '9223372036854775807 + 1', '-(-9223372036854775807 - 1)', '3037000500 * 3037000500',
    '100000000000000000000', 'lib.math.abs(-9223372036854775807 - 1)',
    '9007199254740993 / 1', '-9223372036854775807 % 2 - 9', '-18446744073709551615 % 7',
    'close + lib.math.sin(100000000000000000000)', 'lib.math.max(2 ** 64, 18446744073709551617)',
    '1' + '0' * 400, f'{10 ** 200} * {10 ** 200}', f'-{10 ** 200} * {10 ** 200}',
    str(2 ** 1024 - 2 ** 970), str(2 ** 1024 - 2 ** 970 - 1), f'-{2 ** 1024 - 2 ** 970 - 1}',
    f'{2 ** 1023} + {2 ** 1023}', f'{2 ** 1023} * 2 - {2 ** 1023}', '-0 * 5', '-6 % 3',
]


@pytest.mark.parametrize('source', __test_helper_WIDE_INTS)
def __test_native_const_eval_folds_ints_as_doubles__(source):
    """An int of any size folds natively to Python's value: the double a Pine int is"""
    __test_helper_require_native()
    _native = __test_helper_native()
    node = ast.parse(source, mode='eval').body
    assert __test_helper_same(_native._const_eval(node), __test_helper_python_eval(node))


def __test_native_int_digits_match_python__():
    """An integral double beyond ``i64`` embeds with Python's exact digits"""
    __test_helper_require_native()
    _native = __test_helper_native()
    rnd = random.Random(20261026)
    for _ in range(5_000):
        value = math.ldexp(1.0 + rnd.random(), rnd.randint(53, 1023))
        source = f'{rnd.choice(("", "-"))}{int(value) + rnd.randint(-3, 3)}'
        node = ast.parse(source, mode='eval').body
        assert __test_helper_same(_native._const_eval(node), __test_helper_python_eval(node)), \
            source


def __test_helper_step(source: str):
    """The module after the Python step, and after the native step (or None)."""
    _native = __test_helper_native()
    tree = ast.parse(source)
    context = {'path': str(Path('const_fold_parity.py').resolve()), 'pyne_mode': None,
               'source': source, 'user_code': True, 'host': native._Host()}
    try:
        native_tree = _native.run(tree, ['ConstFold'], context)
    except _native.Unsupported:
        native_tree = None
    python_tree = ConstFoldTransformer().visit(tree)
    return python_tree, native_tree


@pytest.mark.parametrize('name', ['module', 'wildcard'])
def __test_native_const_fold_step_matches_python__(name):
    """The native step folds, stamps and keeps exactly what the Python step does"""
    __test_helper_require_native()
    source = {'module': __test_helper_MODULE, 'wildcard': __test_helper_WILDCARD}[name]
    python_tree, native_tree = __test_helper_step(source)
    assert native_tree is not None, 'the native step declined the module'
    assert native.describe(native_tree) == native.describe(python_tree)


def __test_native_const_fold_carries_a_big_int__():
    """An int beyond 64 bits folds natively as the double it rounds to, bound, read,
    negated and computed with"""
    __test_helper_require_native()
    source = ('A = 18446744073709551615\nB = A\nC = -A\nD = -(-9223372036854775807 - 1)\n'
              'E = 2 ** 70\nF = 1180591620717411303424 + 1\nG = A * A - A\n'
              'H = 9007199254740993 - 9007199254740992\nI = close + 18446744073709551617\n'
              'J = 1' + '0' * 400 + ' + 1\n')
    python_tree, native_tree = __test_helper_step(source)
    assert native_tree is not None, 'the native step declined the module'
    assert native.describe(native_tree) == native.describe(python_tree)
    values = {statement.targets[0].id: statement.value for statement in python_tree.body}
    assert values['A'].value == 18446744073709551616
    assert values['F'].value == 1180591620717411303424
    assert values['H'].value == 0
    assert values['I'].right.value == 18446744073709551616
    assert isinstance(values['J'], ast.BinOp)


def __test_native_round_with_precision_folds_like_python__():
    """``lib.math.round`` with a positive integral precision, ties included, folds to
    Python's value; a fractional one, which scales by ``pine_math.pow``, is left to
    Python"""
    __test_helper_require_native()
    _native = __test_helper_native()
    rnd = random.Random(20261025)
    precisions = ['1', '2', '3', '5', '8', '16', '17', '400', '0.25', '0.75', '1.75', '2.5',
                  '15.99', '16.001']
    sources = [f'lib.math.round({number}, {precision})' for precision in precisions
               for number in ('2.675', '-2.675', '13.887175', '63.986275', '0.149999999996838',
                              '40000.0005', '99999.9995', '1e30', '0.0', '7', '-7')]
    for _ in range(5_000):
        number = rnd.choice((repr(round(rnd.uniform(-1e6, 1e6), rnd.randint(0, 8))),
                             repr(rnd.uniform(-10.0, 10.0)), str(rnd.randint(-10 ** 6, 10 ** 6))))
        precision = rnd.choice((str(rnd.randint(1, 18)), repr(round(rnd.uniform(0.01, 17), 3))))
        sources.append(f'lib.math.round({number}, {precision})')
    for source in sources:
        node = ast.parse(source, mode='eval').body
        precision = __test_helper_python_eval(node.args[1])
        if min(precision, 16) % 1:
            with pytest.raises(_native.Unsupported):
                _native._const_eval(node)
            continue
        got = _native._const_eval(node)
        assert __test_helper_same(got, __test_helper_python_eval(node)), source


def __test_native_integer_round_folds_like_python__():
    """The parser's integer rounding, the number rounded to 15 decimal places first,
    folds natively to Python's value at and around the threshold"""
    __test_helper_require_native()
    _native = __test_helper_native()
    numbers = ['0.4999999999999995', '0.4999999999999994', '0.49999999999999944',
               '0.49999999999999994', '0.25', '0.5', '3.4999999999999996', '2.499999999999999',
               '1.4999999999999996', '1.4999999999999993', '1000.4999999999999',
               '452523.49999999994', '0.49999999999', '4503599627370495.5', '7', '0.0']
    numbers += [repr(0.5 - k * 2.0 ** -54) for k in range(1, 40)]
    numbers += [repr(1.5 - k * 2.0 ** -52) for k in range(1, 10)]
    sources = [f'lib.math.round({sign}{number}{precision})' for number in numbers
               for sign in ('', '-') for precision in ('', ', 0')]
    for source in sources:
        node = ast.parse(source, mode='eval').body
        got = _native._const_eval(node)
        assert __test_helper_same(got, __test_helper_python_eval(node)), source


def __test_native_pow10_is_correctly_rounded__():
    """The power WebAssembly folds ``math.round`` with is the double nearest to 10**y"""
    __test_helper_require_native()
    _native = __test_helper_native()
    rnd = random.Random(20261025)
    exponents = [-rnd.uniform(0.0, 16.0) for _ in range(20_000)]
    exponents += [-k / 1000 for k in range(16_001)]
    packed = _native._pow10(struct.pack(f'<{len(exponents)}d', *exponents))
    powers = struct.unpack(f'<{len(exponents)}d', packed)
    with localcontext() as context:
        context.prec = 60
        for exponent, power in zip(exponents, powers):
            assert power == float(Decimal(10) ** Decimal(exponent)), exponent
