"""
``math.log10`` reproduces the venue's base-10 logarithm, which is NOT correctly rounded.

The venue evaluates a runtime ``math.log10`` with the JVM's x86 ``Math.log10``
intrinsic (the Intel LIBM stub): ``log``'s 129-bin reciprocal reduction, scaled by a
short approximation of ``log10(e)`` in float32, finished by a degree-6 series. Neither
the correctly rounded logarithm nor the platform's ``math.log10`` can stand in for it.

MEASURED on TradingView: 363348 arguments on BINANCE:BTCUSDT@30 (prices, volumes,
ranges and every ratio of two prices of the same bar, probe ``log10_probe``), crafted
arguments for every reciprocal bin (probes ``rcp_probe``/``rcp2``/``rcp3``), and an
independent holdout of 88904 arguments on CAPITALCOM:BTCUSD@15 -- exact on all of them.
"""
import math as _math
import struct
from decimal import Decimal, localcontext

from pynecore.lib import math

# Arguments where macOS's ``math.log10`` differs from the venue, with the venue's value
__test_helper_PLATFORM_DISAGREES = (
    (0.9999999130359312, -3.776801686113221e-08),
    (1.0002773431328527, 0.00012043189248434519),
    (0.9985934771389645, -0.000611275104267816),
    (0.99953823353354, -0.0002005889445006953),
    (1.0006109615387564, 0.0002652562025167785),
    (0.9995094512824626, -0.00021309487211359962),
    (299.00999999999476, 2.4756857129805963),
    (0.997978426875265, -0.0008788466788554364),
)

# Arguments where the venue is NOT the correctly rounded logarithm (191 of the
# 363348 natural ones), so an exact implementation cannot pass either
__test_helper_NOT_CORRECTLY_ROUNDED = (
    (0.9978458884952451, -0.0009365277950983033),
    (0.9989204236308412, -0.0004691073240957333),
    (0.9976999285332933, -0.0010000588908282352),
    (0.9999999130359312, -3.776801686113221e-08),
    (1.0002773431328527, 0.00012043189248434519),
)

# Arguments in the reciprocal bins where the processor's approximate reciprocal is
# not the textbook one (1/midpoint rounded to 12 bits): a reduction built on the
# textbook reciprocal picks the neighbouring table entry and misses these
__test_helper_MEASURED_RECIPROCAL = (
    (1.0023789153935747, 0.0010319228858053377),
    (1.002274783982184, 0.000986804172867973),
    (1.0024287520807198, 0.001053514780707831),
    (1.0023265568459483, 0.0010092372308453698),
    (1.0022861039217348, 0.0009917091745689681),
    (1.0301398079893274, 0.012896170062325739),
    (0.5151103747869467, -0.2880997029492658),
    (1.0300963994121477, 0.012877869145691558),
)


def __test_helper_exact(x: float) -> float:
    with localcontext() as ctx:
        ctx.prec = 60
        return float(Decimal(x).log10())


def __test_helper_ulps(a: float, b: float) -> int:
    """Signed lattice distance between two finite doubles of the same sign."""
    ia = struct.unpack('<q', struct.pack('<d', a))[0]
    ib = struct.unpack('<q', struct.pack('<d', b))[0]
    if ia < 0:
        ia = -(ia & 0x7FFFFFFFFFFFFFFF)
    if ib < 0:
        ib = -(ib & 0x7FFFFFFFFFFFFFFF)
    return abs(ia - ib)


def __test_log10_matches_the_venue_where_macos_does_not__():
    for x, expected in __test_helper_PLATFORM_DISAGREES:
        assert math.log10(x) == expected, x.hex()

    # The venue is not the correctly rounded logarithm on most of them either, which
    # holds whatever the host's libm is
    assert sum(1 for x, e in __test_helper_PLATFORM_DISAGREES
               if __test_helper_exact(x) != e) == 5


def __test_log10_matches_the_venue_where_it_is_not_correctly_rounded__():
    for x, expected in __test_helper_NOT_CORRECTLY_ROUNDED:
        assert math.log10(x) == expected, x.hex()

    assert all(__test_helper_exact(x) != e for x, e in __test_helper_NOT_CORRECTLY_ROUNDED)


def __test_log10_matches_the_venue_where_the_reciprocal_decides__():
    for x, expected in __test_helper_MEASURED_RECIPROCAL:
        assert math.log10(x) == expected, x.hex()

    assert sum(1 for x, e in __test_helper_MEASURED_RECIPROCAL
               if __test_helper_exact(x) != e) == 6


def __test_log10_stays_within_an_ulp_over_the_price_range__():
    x = 1e-6
    while x < 1e9:
        assert __test_helper_ulps(math.log10(x), __test_helper_exact(x)) <= 1, x
        x *= 1.0000313


def __test_log10_handles_the_extremes__():
    for bits in (1, 2, 0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff):
        x = struct.unpack('<d', struct.pack('<Q', bits))[0]
        assert __test_helper_ulps(math.log10(x), __test_helper_exact(x)) <= 1, x.hex()


def __test_log10_edge_values__():
    assert math.log10(1.0) == 0.0
    assert math.log10(10.0) == 1.0
    assert math.log10(1000.0) == 3.0
    assert math.log10(0.0) == float('-inf')
    assert math.log10(-0.0) == float('-inf')
    assert _math.isnan(math.log10(-1.0))
    assert _math.isnan(math.log10(float('-inf')))
    assert math.log10(float('inf')) == float('inf')
    assert math.log10(float('nan')) != math.log10(float('nan'))
