"""
``math.pow`` reproduces the venue's power function, which is NOT correctly rounded.

The venue evaluates a runtime ``math.pow`` with the JVM's x86 ``Math.pow`` intrinsic
(the Intel LIBM stub): ``log2(x)`` in head and tail halves from a 513-bin reciprocal
reduction, then ``2**(y * log2(x))`` from a 256-entry table, with separate paths for
a base near 1 against a large exponent, for results near the range limits and for
negative bases. The platform's ``pow()`` behind Python's ``**`` is a different
algorithm and misses the venue's last bit on roughly one argument in 700.

MEASURED on TradingView: 423906 arguments on BINANCE:BTCUSDT@30 (14 exponent
patterns over prices, ratios and volumes, probe ``pow_probe``), crafted arguments
for every reciprocal bin (probes ``rcp2``/``rcp3``), and an independent holdout of
244486 arguments on CAPITALCOM:BTCUSD@15 -- exact on all of them.
"""
import math as _math

from pynecore.lib import math

# (base, exponent, venue) where CPython's ``**`` differs from the venue, over the
# measured exponent patterns: fractional and integer, positive and negative, a
# variable exponent, a constant base of 10 and 2
__test_helper_PLATFORM_DISAGREES = (
    (1.002961900811646, -7.25, 0.9787862043785036),
    (0.9987166934492674, -7.25, 1.0093534198949032),
    (2.0, -1.0636000000000059, 0.4784367114813817),
    (2.0, -0.6944999999999709, 0.6179234374015851),
    (0.999195324094812, 2.5, 0.9979895241429024),
    (63959.93, -0.3333, 0.025014444930144944),
    (1.0021071970164595, 10.0, 1.0212729096676236),
    (10.0, 0.9768128841898206, 9.480099251503457),
    (10.0, 1.0033943349596302, 10.07846367741895),
    (684.28203, 0.7, 96.53050863533902),
    (103648.32, 1.5, 33369016.042688776),
    (1.0044781007782928, 0.3333333333333333, 1.0014904776321485),
    (93130.04, 1.000907492741026, 94102.07553031585),
    (0.9972583252017253, 365.0, 0.3671129816522114),
    (1.0045709010659836, 111.90136, 1.6658318537910928),
    (0.9942017631430812, 18.093813253542894, 0.9001286794949037),
)

# ``pow(x, 0.5)`` is ``sqrt(x)`` on the venue; the platform ``pow()`` is not
__test_helper_HALF_IS_SQRT = (
    (67958.52, 260.6885498060857),
    (102120.01, 319.56221616455224),
)

# A base within 1/16 of 1 against an exponent of at least 2**12 takes the stub's
# extra-precision logarithm
__test_helper_NEAR_ONE = (
    (1.002175764778405, 5000.0, 52418.04334825068),
    (1.0009389141018297, 5000.0, 109.11124831269824),
    (0.9999597259875523, 5000.0, 0.8176064935646796),
    (0.9990271500656545, 5000.0, 0.007699365932792211),
)

# A negative base with an odd integer exponent: the sign comes from the exponent's
# parity, the magnitude from the positive path
__test_helper_NEGATIVE_BASE = (
    (-0.9898623904991166, 3.0, -0.9698944430229797),
    (-0.9969975322603941, 3.0, -0.9910196141520806),
    (-0.9975830699224655, 3.0, -0.9927667203017753),
    (-0.9992489826929962, 3.0, -0.997748639736381),
)

# Arguments in the reciprocal bins where the processor's approximate reciprocal is
# not the textbook one: a reduction built on the textbook reciprocal picks the
# neighbouring table row and misses these
__test_helper_MEASURED_RECIPROCAL = (
    (8.016931407509047, 25.0, 3.982943953744096e+22),
    (0.06366963599900295, 77.7, 1.162906815282943e-93),
    (2.053868451106672, -123.456, 2.57652372561939e-39),
    (2.0626376599416827, 77.7, 2.695644786003091e+24),
    (0.2698855815920854, 365.0, 2.402063186203353e-208),
    (0.27302572118239216, 25.0, 8.03690941894409e-15),
    (1.1037595212208207, -3000.25, 2.3221628912305492e-129),
    (1.1102217323371693, 77.7, 3375.480299846336),
)


def __test_pow_matches_the_venue_where_the_platform_does_not__():
    for x, y, expected in __test_helper_PLATFORM_DISAGREES:
        assert math.pow(x, y) == expected, (x, y)

    # The platform is what differs -- otherwise the test proves nothing
    assert all(x ** y != e for x, y, e in __test_helper_PLATFORM_DISAGREES)


def __test_pow_half_is_sqrt__():
    for x, expected in __test_helper_HALF_IS_SQRT:
        assert math.pow(x, 0.5) == expected == _math.sqrt(x)
        assert x ** 0.5 != expected


def __test_pow_matches_the_venue_near_one__():
    for x, y, expected in __test_helper_NEAR_ONE:
        assert math.pow(x, y) == expected, (x, y)
    assert all(x ** y != e for x, y, e in __test_helper_NEAR_ONE)


def __test_pow_matches_the_venue_on_a_negative_base__():
    for x, y, expected in __test_helper_NEGATIVE_BASE:
        assert math.pow(x, y) == expected, (x, y)
    assert all(x ** y != e for x, y, e in __test_helper_NEGATIVE_BASE)


def __test_pow_matches_the_venue_where_the_reciprocal_decides__():
    for x, y, expected in __test_helper_MEASURED_RECIPROCAL:
        assert math.pow(x, y) == expected, (x, y)


def __test_pow_edge_values__():
    inf = float('inf')
    assert math.pow(2.0, 10.0) == 1024.0
    assert math.pow(-2.0, 3.0) == -8.0
    assert math.pow(-2.0, 2.0) == 4.0
    assert _math.isnan(math.pow(-8.0, 1.0 / 3.0))
    assert math.pow(0.0, -1.0) == inf
    assert math.pow(-0.0, -1.0) == -inf
    assert math.pow(-0.0, -2.0) == inf
    assert math.pow(0.0, 3.0) == 0.0
    assert math.pow(inf, -1.0) == 0.0
    assert math.pow(-inf, 3.0) == -inf
    assert math.pow(-inf, 2.0) == inf
    assert math.pow(0.5, inf) == 0.0
    assert math.pow(0.5, -inf) == inf
    assert math.pow(2.0, inf) == inf
    # |x| == 1 against an infinite exponent is NaN on the JVM, not 1
    assert _math.isnan(math.pow(1.0, inf))
    assert _math.isnan(math.pow(-1.0, -inf))
    # beyond the range: overflow and underflow, a subnormal result on the way
    assert math.pow(10.0, 400.0) == inf
    assert math.pow(10.0, -400.0) == 0.0
    assert 0.0 < math.pow(10.0, -310.0) < 2.2250738585072014e-308
    assert math.pow(float('nan'), 0.0) != math.pow(float('nan'), 0.0)
