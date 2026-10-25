"""
The native fdlibm port (``rust/crates/pyne-transform/src/fdlibm.rs``) returns the very
doubles ``core.fdlibm`` does, and the native ``quantize_embed`` the very doubles of
``transformers.const_fold``.

The native constant fold computes its transcendentals with that port, so a single
disagreeing argument would embed a different literal. The arguments are the specials,
seeded random bit patterns and ranges reaching every branch: subnormals, huge and tiny
magnitudes, the exp overflow and underflow edges, arguments next to multiples of pi/2
(the argument reduction's cancellations), values next to one (log, pow).

The default run checks ``__test_helper_DEFAULT`` random arguments per function;
``PYNE_FDLIBM_SWEEP=<count>`` sets the count, e.g. ``2000000`` for the full sweep.
"""
import math
import os
import random
import struct
from array import array

import pytest

from pynecore.core import fdlibm
from pynecore.transformers.const_fold import quantize_embed

from .test_137_native_transform import __test_helper_require_native

#: Random arguments per function in the default run
__test_helper_DEFAULT = 20_000

__test_helper_UNARY = ('sin', 'cos', 'tan', 'exp', 'log', 'asin', 'acos', 'atan')

__test_helper_SPECIALS = (
    0.0, -0.0, math.inf, -math.inf, math.nan, -math.nan, 5e-324, -5e-324,
    2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308,
    -1.7976931348623157e308, 1.0, -1.0, 0.5, -0.5, 1e-300, -1e-300, 709.78,
    709.782712893384, 709.7827128933841, -708.4, -730.0, -744.44, -745.1332191019411,
    -745.2, 90111.9, 90112.0, -90112.0, 1e6, 1e22, -1e22, 1e300, 0.785, 2.356,
    3.141592653589793, 1.5707963267948966, 0.7853981633974483, 2.356194490192345,
    843314856.0, 1.4e19, 2.0 ** 19 * math.pi / 2, 2.0 ** 66, -2.0 ** 66, 0.4375, 0.6875,
    1.1875, 2.4375, 0.6744, 0.975, 0.3, 0.78125, 2.0 ** -27, 2.0 ** -28, 2.0 ** -29,
    2.0 ** -57, 1.0 + 2.0 ** -20, 1.0 - 2.0 ** -21, 10.0, 0.1, 1e-310, 3.0, 1.5,
)


def __test_helper_count() -> int:
    return int(os.environ.get('PYNE_FDLIBM_SWEEP', __test_helper_DEFAULT))


def __test_helper_float(bits: int) -> float:
    return struct.unpack('<d', struct.pack('<Q', bits & 0xFFFFFFFFFFFFFFFF))[0]


def __test_helper_bits(x: float) -> int:
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def __test_helper_nudge(x: float, ulps: int) -> float:
    """``x`` moved by ``ulps`` units in the last place, through the bit pattern."""
    if not math.isfinite(x) or x == 0.0:
        return x
    return __test_helper_float(__test_helper_bits(x) + ulps)


def __test_helper_arguments(count: int, seed: int) -> list[float]:
    """Specials plus seeded random arguments reaching every branch of the nine functions."""
    rnd = random.Random(seed)
    args = list(__test_helper_SPECIALS)
    for _ in range(count):
        pick = rnd.random()
        sign = rnd.choice((1.0, -1.0))
        if pick < 0.2:
            args.append(__test_helper_float(rnd.getrandbits(64)))
        elif pick < 0.3:
            args.append(rnd.uniform(-800.0, 800.0))
        elif pick < 0.4:
            args.append(rnd.uniform(-10.0, 10.0))
        elif pick < 0.5:
            args.append(rnd.uniform(-1.2, 1.2))
        elif pick < 0.55:
            args.append(1.0 + rnd.uniform(-0.02, 0.02))
        elif pick < 0.6:
            args.append(1.0 + sign * math.ldexp(rnd.random(), -rnd.randint(1, 60)))
        elif pick < 0.65:
            args.append(rnd.uniform(-750.0, -700.0))
        elif pick < 0.7:
            # Subnormals
            args.append(sign * __test_helper_float(rnd.randint(1, (1 << 52) - 1)))
        elif pick < 0.85:
            # Next to a multiple of pi/2, small and large multiples alike
            k = rnd.randint(1, 1 << rnd.choice((4, 10, 20, 30, 52, 64)))
            x = sign * k * (math.pi / 2)
            args.append(__test_helper_nudge(x, rnd.randint(-16, 16)))
        else:
            args.append(sign * math.ldexp(rnd.random(), rnd.randint(-1074, 1024)))
    return args


def __test_helper_exponents(count: int, seed: int) -> list[float]:
    """Exponents for ``pow``: its shortcuts, integers of both parities, fractions, and
    magnitudes reaching every range branch (near one, wide, overflow, underflow)."""
    rnd = random.Random(seed)
    exps = [2.0, 0.5, 1.0, -1.0, 3.0, -2.0, 0.0, -0.0, math.inf, -math.inf, math.nan,
            1e300, -1e300, 5e-324, 1.0 / 3.0, 2.0 ** 52, 2.0 ** 53 + 2.0, 2.0 ** 53 - 1.0,
            2.0 ** 31 + 1.0, 2.0 ** 64 + 2.0 ** 12, -(2.0 ** 31) - 1.0, 1075.0, -1075.0, 1024.0]
    while len(exps) < count:
        pick = rnd.random()
        if pick < 0.2:
            exps.append(__test_helper_float(rnd.getrandbits(64)))
        elif pick < 0.4:
            exps.append(float(rnd.randint(-40, 40)))
        elif pick < 0.6:
            exps.append(rnd.uniform(-5.0, 5.0))
        elif pick < 0.7:
            exps.append(float(rnd.randint(-(1 << 40), 1 << 40)))
        elif pick < 0.8:
            exps.append(rnd.uniform(-1100.0, 1100.0))
        else:
            exps.append(math.ldexp(rnd.random(), rnd.randint(-60, 70)) * rnd.choice((1.0, -1.0)))
    return exps[:count]


def __test_helper_native(name: str, xs: list[float], ys: list[float] | None = None) -> array:
    from pynecore.transformers import _native

    out = array('d')
    out.frombytes(_native._fdlibm(name, array('d', xs).tobytes(),
                                  None if ys is None else array('d', ys).tobytes()))
    return out


def __test_helper_reference(name: str):
    """The pure-Python original (the public name may be the compiled twin)."""
    return getattr(fdlibm, 'PYTHON_IMPLEMENTATIONS', {}).get(name, getattr(fdlibm, name))


def __test_helper_mismatches(got: array, want: list[float], args) -> list[str]:
    return [f'{arg!r}: native {g!r}, Python {w!r}'
            for arg, g, w in zip(args, got, want)
            if __test_helper_bits(g) != __test_helper_bits(w)]


@pytest.mark.parametrize('name', __test_helper_UNARY)
def __test_native_fdlibm_matches_python__(name):
    """Every native fdlibm function agrees with the Python port bit for bit"""
    __test_helper_require_native()
    args = __test_helper_arguments(__test_helper_count(), 20261022)
    reference = __test_helper_reference(name)
    want = [reference(x) for x in args]
    bad = __test_helper_mismatches(__test_helper_native(name, args), want, args)
    assert not bad, f'{name}: {len(bad)} of {len(args)} differ, e.g. {bad[:5]}'


def __test_native_fdlibm_pow_matches_python__():
    """The native ``pow`` agrees with the Python port bit for bit"""
    __test_helper_require_native()
    count = __test_helper_count()
    args = __test_helper_arguments(count, 20261023)
    exps = __test_helper_exponents(len(args), 20261024)
    rnd = random.Random(20261025)
    bases = []
    for x in args:
        near_one = 1.0 + rnd.uniform(-0.0625, 0.0625) * rnd.choice((1.0, 2.0 ** -20, 2.0 ** -45))
        bases.append(x if rnd.random() < 0.5 else near_one)
    reference = __test_helper_reference('pow')
    want = [reference(x, y) for x, y in zip(bases, exps)]
    bad = __test_helper_mismatches(__test_helper_native('pow', bases, exps), want,
                                   list(zip(bases, exps)))
    assert not bad, f'pow: {len(bad)} of {len(bases)} differ, e.g. {bad[:5]}'


def __test_helper_embed_values(count: int, seed: int) -> list[float]:
    """Doubles for ``quantize_embed``: mostly in [1e-3, 1), where the 16-place cap
    rounds, ties at the 17th decimal among them, plus any double at all."""
    rnd = random.Random(seed)
    values = [0.21402930929524275, 0.27161597376132945, 1e-3, 0.001, 0.0009999999999999998,
              -0.0, 0.0, 1.0, 0.1, 1.0 / 3.0, 2.0 / 3.0, 0.12345678901234567, 1e300, 5e-324]
    for _ in range(count):
        pick = rnd.random()
        sign = rnd.choice((1.0, -1.0))
        if pick < 0.3:
            values.append(sign * rnd.uniform(1e-3, 1.0))
        elif pick < 0.6:
            # A decimal tie at the 17th place, spelled with 17 to 19 places
            digits = ''.join(rnd.choice('0123456789') for _ in range(16))
            tail = rnd.choice(('5', '50', '500', '49', '51', '499', '501'))
            head = rnd.choice(('0.', '0.0', '0.00'))
            values.append(sign * float(head + digits[:16 - len(head) + 2] + tail))
        elif pick < 0.8:
            values.append(sign * math.ldexp(rnd.random(), rnd.randint(-12, 12)))
        else:
            value = __test_helper_float(rnd.getrandbits(64))
            if math.isfinite(value):
                values.append(value)
    return values


def __test_native_quantize_embed_matches_python__():
    """The native ``quantize_embed`` embeds the very double the Python one does"""
    __test_helper_require_native()
    from pynecore.transformers import _native

    values = __test_helper_embed_values(__test_helper_count(), 20261026)
    got = array('d')
    got.frombytes(_native._quantize_embed(array('d', values).tobytes()))
    want = [quantize_embed(value) for value in values]
    bad = __test_helper_mismatches(got, want, values)
    assert not bad, f'quantize_embed: {len(bad)} of {len(values)} differ, e.g. {bad[:5]}'
