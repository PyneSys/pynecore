"""
``math.log`` reproduces the venue's logarithm, which is NOT correctly rounded.

The venue evaluates a runtime ``math.log`` with the JVM's x86 ``Math.log``
intrinsic (the Intel LIBM stub): a 129-bin reciprocal reduction finished by a
degree-7 ``log1p`` series in plain doubles. That is accurate to well under an
ulp but not correctly rounded, so neither the correctly rounded logarithm nor
the platform's ``math.log`` can stand in for it.

MEASURED on TradingView, BINANCE:BTCUSDT@30 with byte-identical bar data:
``log(high)``/``log(low)``/``log(close)`` over 86241 values (probe ``logpow``)
and, over 30213 bars, the logarithm of every ratio of two prices of the same bar
-- 120856 arguments (probe ``rs_probe``). Away from 1 the venue happens to be
correctly rounded on every value seen; within about 1.2% of 1 it is not, and
those are the arguments a log-return script spends all its time on.
"""
import math as _math
import struct
from decimal import Decimal, localcontext

from pynecore.lib import math


# Arguments where macOS's ``math.log`` differs from the venue, with the
# venue's value. Here the venue agrees with the correctly rounded logarithm.
_TV_DISAGREEING = (
    (101976.4, 11.532496692946632),
    (117512.02, 11.674295905206622),
    (88719.06, 11.39323002683737),
    (68826.2, 11.139339765383694),
    (73069.4, 11.19916495344732),
    (77532.23, 11.258448999858816),
)

# Every argument of the 30213-bar ``log(close / open)`` run where the venue and
# a correctly rounded logarithm land on different doubles -- the whole reason the
# stub is ported operation for operation instead of computing an exact one. On
# most of them it is the venue that is the ulp out.
_TV_NEAR_ONE = (
    (1.0024528394938568, 0.0024498361931393404),
    (0.9961823527585483, -0.0038249530566260973),
    (1.0016271269933827, 0.0016258046564688428),
    (1.0005515804040028, 0.0005514283394464373),
    (0.9963816236118358, -0.0036249385463619443),
    (0.996197708529301, -0.003809538557102234),
    (0.998856003265845, -0.0011446515979078458),
    (1.0017897554053532, 0.0017881557015822776),
    (0.995554660727645, -0.004455249172462254),
    (0.9983923970854343, -0.0016088964946925448),
    (0.996628420984103, -0.0033772755963453523),
    (0.9995060577244973, -0.0004940643051738624),
)


# Arguments where the bin reduction's APPROXIMATE reciprocal is what decides the
# last bit: dividing instead picks the neighbouring bin here, because the exact
# reciprocal sits just under the boundary the half-bin carry rounds across and
# the instruction's own lean reaches over it. Roughly one argument in 20000.
_TV_APPROXIMATE_RECIPROCAL = (
    (0.9885051088405153, -0.01156146810913521),
    (0.9962033834899373, -0.0038038419524787466),
    (0.9962897780443103, -0.003717121901343816),
    (0.9962532979380777, -0.003753738531294136),
    (0.9962690777772613, -0.003737899472846208),
    (0.9963109097206667, -0.003695911754737957),
    (1.002007332005773, 0.002005320006932131),
)


def _exact(x: float) -> float:
    with localcontext() as ctx:
        ctx.prec = 50
        return float(Decimal(x).ln())


def _ulps(a: float, b: float) -> int:
    """Signed lattice distance between two finite doubles of the same sign."""
    ia = struct.unpack('<q', struct.pack('<d', a))[0]
    ib = struct.unpack('<q', struct.pack('<d', b))[0]
    if ia < 0:
        ia = -(ia & 0x7FFFFFFFFFFFFFFF)
    if ib < 0:
        ib = -(ib & 0x7FFFFFFFFFFFFFFF)
    return abs(ia - ib)


def __test_log_matches_the_venue_where_macos_does_not__():
    for x, expected in _TV_DISAGREEING:
        assert math.log(x) == expected


def __test_log_matches_the_venue_near_one__():
    for x, expected in _TV_NEAR_ONE:
        assert math.log(x) == expected, x.hex()

    # On most of these the venue is the side that is an ulp out, so a correctly
    # rounded implementation cannot pass this test either.
    assert sum(1 for x, e in _TV_NEAR_ONE if _exact(x) != e) >= 8


def __test_log_matches_the_venue_where_the_reciprocal_decides__():
    for x, expected in _TV_APPROXIMATE_RECIPROCAL:
        assert math.log(x) == expected, x.hex()

    assert sum(1 for x, e in _TV_APPROXIMATE_RECIPROCAL if _exact(x) != e) == 6


def __test_log_stays_within_an_ulp_over_the_price_range__():
    x = 1e-6
    while x < 1e9:
        assert _ulps(math.log(x), _exact(x)) <= 1, x
        x *= 1.0000313


def __test_log_stays_within_an_ulp_near_one__():
    """Where the bin reduction is weakest and the venue's own ulp shows."""
    for i in range(-20000, 20000, 7):
        x = 1.0 + i * 1e-7
        assert _ulps(math.log(x), _exact(x)) <= 1, x


def __test_log_handles_the_extremes__():
    for bits in (1, 2, 0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff):
        x = struct.unpack('<d', struct.pack('<Q', bits))[0]
        assert _ulps(math.log(x), _exact(x)) <= 1, x.hex()


def __test_log_edge_values__():
    assert math.log(1.0) == 0.0
    assert math.log(_math.e) == 1.0
    assert math.log(0.0) == float('-inf')
    assert math.log(-0.0) == float('-inf')
    assert _math.isnan(math.log(-1.0))
    assert _math.isnan(math.log(float('-inf')))
    assert math.log(float('inf')) == float('inf')
    assert math.log(float('nan')) != math.log(float('nan'))
