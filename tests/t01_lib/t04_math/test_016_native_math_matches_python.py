"""
The compiled transcendental functions return the very doubles the Python ones do.

``core/_native_math`` is a compiled copy of ``core/pine_math`` and ``core/fdlibm``, and
it only earns its place if it is bit-identical to them on every argument: those Python
modules are what the venue tests pin, and they stay the implementation wherever the
extension is not built. The copy is exact only when the C compiler keeps every
operation as written -- built WITHOUT ``-ffp-contract=off``, clang fuses multiply-adds
and this test finds hundreds of disagreeing arguments per function.

The wheel build runs this test on every platform it produces a wheel for, with
``PYNE_REQUIRE_NATIVE_MATH=1`` so that a wheel whose extension silently failed to build
fails instead of skipping.
"""
import math as _pymath
import os
import random
import struct

import pytest

from pynecore.core import fdlibm, pine_math

_FUNCTIONS = (
    (pine_math, ('cos', 'sin', 'exp', 'log', 'log10')),
    (fdlibm, ('sin', 'cos', 'exp', 'asin', 'acos', 'atan')),
)

_SPECIALS = (
    0.0, -0.0, _pymath.inf, -_pymath.inf, _pymath.nan, 5e-324, -5e-324,
    2.2250738585072014e-308, 1.7976931348623157e308, -1.7976931348623157e308,
    1.0, -1.0, 0.5, -0.5, 1e-300, -1e-300, 709.78, 709.782712893384, -708.4, -730.0,
    -744.44, -745.2, 90111.9, 90112.0, -90112.0, 1e6, 1e22, 1e300, 0.785, 2.356,
    3.141592653589793, 1.5707963267948966, 843314856.0, 1.4e19,
)


def __test_helper_bits(x: float) -> int:
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def __test_helper_arguments(count: int) -> list[float]:
    """Specials plus seeded random arguments covering every branch of the ten functions."""
    rnd = random.Random(20260923)
    args = list(_SPECIALS)
    for _ in range(count):
        pick = rnd.random()
        if pick < 0.2:
            args.append(struct.unpack('<d', struct.pack('<Q', rnd.getrandbits(64)))[0])
        elif pick < 0.4:
            args.append(rnd.uniform(-800.0, 800.0))
        elif pick < 0.55:
            args.append(rnd.uniform(-10.0, 10.0))
        elif pick < 0.65:
            args.append(1.0 + rnd.uniform(-0.02, 0.02))
        elif pick < 0.8:
            args.append(rnd.uniform(-1.2, 1.2))
        elif pick < 0.9:
            args.append(rnd.uniform(-750.0, -700.0))
        else:
            args.append(_pymath.ldexp(rnd.random(), rnd.randint(-1074, 1024))
                        * rnd.choice((1.0, -1.0)))
    return args


def __test_helper_exponents(count: int) -> list[float]:
    """Exponents for ``pow``: the stub's shortcuts, integers of both parities, fractions,
    and magnitudes that reach every range branch (near one, wide, overflow, underflow)."""
    rnd = random.Random(20260924)
    exps = [2.0, 0.5, 1.0, -1.0, 3.0, -2.0, 0.0, -0.0, _pymath.inf, -_pymath.inf, _pymath.nan,
            1e300, -1e300, 5e-324, 1.0 / 3.0, 2.0 ** 52, 2.0 ** 53 + 2.0, 2.0 ** 53 - 1.0]
    for _ in range(count):
        pick = rnd.random()
        if pick < 0.2:
            exps.append(struct.unpack('<d', struct.pack('<Q', rnd.getrandbits(64)))[0])
        elif pick < 0.4:
            exps.append(float(rnd.randint(-40, 40)))
        elif pick < 0.7:
            exps.append(rnd.uniform(-5.0, 5.0))
        else:
            exps.append(_pymath.ldexp(rnd.random(), rnd.randint(-60, 20))
                        * rnd.choice((1.0, -1.0)))
    return exps


def __test_native_math_matches_python__():
    """Every compiled function agrees with its Python original bit for bit"""
    native = pine_math.log is not pine_math.PYTHON_IMPLEMENTATIONS['log']
    if not native:
        if os.environ.get('PYNE_REQUIRE_NATIVE_MATH'):
            pytest.fail("the native math extension is not installed")
        pytest.skip("native math extension not built")

    args = __test_helper_arguments(100_000)
    for module, names in _FUNCTIONS:
        for name in names:
            compiled = getattr(module, name)
            original = module.PYTHON_IMPLEMENTATIONS[name]
            assert compiled is not original, f"{module.__name__}.{name} is not compiled"
            for x in args:
                got, want = compiled(x), original(x)
                assert __test_helper_bits(got) == __test_helper_bits(want), (
                    f"{module.__name__}.{name}({x!r}): compiled {got!r}, Python {want!r}")

    compiled, original = pine_math.pow, pine_math.PYTHON_IMPLEMENTATIONS['pow']
    assert compiled is not original, "pine_math.pow is not compiled"
    exps = __test_helper_exponents(len(args))
    rnd = random.Random(20260925)
    for x, y in zip(args, exps):
        near_one = 1.0 + rnd.uniform(-0.0625, 0.0625) * rnd.choice((1.0, 2.0 ** -20, 2.0 ** -45))
        for base in (x, near_one):
            got, want = compiled(base, y), original(base, y)
            assert __test_helper_bits(got) == __test_helper_bits(want), (
                f"pine_math.pow({base!r}, {y!r}): compiled {got!r}, Python {want!r}")
