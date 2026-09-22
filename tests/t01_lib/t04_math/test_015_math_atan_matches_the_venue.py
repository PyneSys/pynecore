"""
``math.atan`` reproduces the venue's arc tangent.

The venue's JVM has no JIT intrinsic for ``atan``: ``Math.atan`` delegates to
``StrictMath.atan``, the fdlibm algorithm, which reduces the argument into one of
five ranges and finishes with a degree-11 odd/even split polynomial. It is not
correctly rounded, and the platform's ``math.atan`` is not it.

MEASURED on TradingView, BINANCE:BTCUSDT@30 with byte-identical bar data (probe
``atan_probe``, 47116 values covering every reduction branch): the fdlibm port is
exact on all of them, the platform's ``atan`` misses 3216 -- 6.8%, by an ulp each.
"""
import math as _pymath

from pynecore.lib import math


# Arguments where the platform's ``atan`` differs from the venue, with the
# venue's value: two per reduction branch where the run offers two.
_TV_DISAGREEING = (
    # |x| < 0.4375, no reduction
    (0.18741376536792875, 0.1852646427859228),
    (-0.274727722527673, -0.2681130590903734),
    # 7/16 <= |x| < 11/16
    (0.4598856377106222, 0.43104434714436146),
    (-0.5695798967055514, -0.5177513881273137),
    # 11/16 <= |x| < 19/16
    (0.8980997236441775, 0.7317642327789431),
    (-0.7204181090742852, -0.6242983600010581),
    # 19/16 <= |x| < 2.4375
    (2.1384019351682153, 1.1333710703482753),
    # 2.4375 <= |x| < 2**66
    (6.840860561710075, 1.4256439680494677),
)


def __test_atan_matches_the_venue_where_the_platform_does_not__():
    for x, expected in _TV_DISAGREEING:
        assert math.atan(x) == expected, x.hex()

    # The platform is what differs -- otherwise the test proves nothing about
    # the code under test.
    assert all(_pymath.atan(x) != expected for x, expected in _TV_DISAGREEING)


def __test_atan_is_odd__():
    for x, _ in _TV_DISAGREEING:
        assert math.atan(-x) == -math.atan(x)


def __test_atan_edge_values__():
    assert math.atan(0.0) == 0.0
    assert math.atan(-0.0) == -0.0
    assert math.atan(float('inf')) == _pymath.pi / 2
    assert math.atan(float('-inf')) == -_pymath.pi / 2
    assert math.atan(2.0 ** 70) == _pymath.pi / 2
    assert math.atan(1e-300) == 1e-300
    assert math.atan(float('nan')) != math.atan(float('nan'))
