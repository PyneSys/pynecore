"""
``math.log`` is correctly rounded, like the venue's.

MEASURED on TradingView (BINANCE:BTCUSDT@30, probe ``logpow``, 86241 values of
``log(high)``/``log(low)``/``log(close)``): every one of them equals the
correctly rounded natural logarithm. The platform ``math.log`` behind CPython
is only within an ulp, and differed on 7 distinct arguments of that run -- a
recursive script (a log-scale envelope, a log-return regime) carries that ulp
into its plot, so the runtime cannot use it.
"""
import math as _math
import struct
from decimal import Decimal, localcontext

from pynecore.lib import math


# Arguments where CPython's ``math.log`` differs from the venue on this
# platform, with the venue's value.
_TV_DISAGREEING = (
    (101976.4, 11.532496692946632),
    (117512.02, 11.674295905206622),
    (88719.06, 11.39323002683737),
    (68826.2, 11.139339765383694),
    (73069.4, 11.19916495344732),
    (77532.23, 11.258448999858816),
)


def _exact(x: float) -> float:
    with localcontext() as ctx:
        ctx.prec = 50
        return float(Decimal(x).ln())


def __test_log_matches_the_venue_where_the_platform_does_not__():
    for x, expected in _TV_DISAGREEING:
        assert math.log(x) == expected

    # The platform is what differs -- otherwise the test proves nothing about
    # the code under test.
    assert all(_math.log(x) != expected for x, expected in _TV_DISAGREEING)


def __test_log_is_correctly_rounded_over_the_price_range__():
    x = 1e-6
    while x < 1e9:
        assert math.log(x) == _exact(x), x
        x *= 1.0000313


def __test_log_is_correctly_rounded_near_one__():
    """The reduction-free branch: ``x - 1`` is exact, the series carries it."""
    for i in range(-20000, 20000, 7):
        x = 1.0 + i * 1e-7
        assert math.log(x) == _exact(x), x


def __test_log_handles_the_extremes__():
    for bits in (1, 2, 0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff):
        x = struct.unpack('<d', struct.pack('<Q', bits))[0]
        assert math.log(x) == _exact(x), x.hex()


def __test_log_edge_values__():
    assert math.log(1.0) == 0.0
    assert math.log(_math.e) == 1.0
    assert math.log(0.0) == float('-inf')
    assert _math.isnan(math.log(-1.0))
    assert math.log(float('inf')) == float('inf')
    assert math.log(float('nan')) != math.log(float('nan'))

def __test_log_is_correctly_rounded_at_power_of_two_results__():
    """A result on a binade edge has an asymmetric rounding boundary.

    Below ``2**k`` the gap is half of what it is above, so a Ziv test that
    measures the remainder against a single ulp mis-decides the arguments whose
    logarithm lands just inside the narrow side.
    """
    for k in range(-40, 10):
        for sign in (1.0, -1.0):
            try:
                x = _math.exp(sign * 2.0 ** k)
            except OverflowError:
                continue
            if x <= 0.0 or not _math.isfinite(x):
                continue
            for _ in range(30):
                x = _math.nextafter(x, 0.0)
            for _ in range(60):
                if x > 0.0:
                    assert math.log(x) == _exact(x), x.hex()
                x = _math.nextafter(x, _math.inf)
