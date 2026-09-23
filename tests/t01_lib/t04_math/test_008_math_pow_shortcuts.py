"""
``math.pow`` takes TradingView's exact shortcuts for four exponents.

MEASURED on TradingView (BINANCE:BTCUSDT@30, 8000 bars, base ``x`` in
[0.3, 1.3] built from ``ta.correlation``): ``pow(x, 2) - x * x``,
``pow(x, 0.5) - sqrt(x)``, ``pow(x, 1) - x`` and ``pow(x, 0) - 1`` were zero on
EVERY bar, while ``pow(x, -1) - 1 / x`` was non-zero on 3 and
``pow(x, 3) - x * x * x`` on 2149 -- so only these four are shortcuts. All four
fall out of the venue's ``pow`` stub itself (``core/pine_math.pow``), which tests
``y == 2`` and ``y == 0.5`` first and is exact on ``y == 1`` and ``y == 0``.

macOS's ``pow()`` behind Python's ``**`` disagrees with the shortcut on a
handful of those bases, which a recursive script turns into a visible plot
divergence, so the bases below are exactly the ones measured to disagree.
"""
import math as _math

from pynecore.lib import math


# Bases where macOS's ``pow()`` differs from the shortcut (a correctly rounding
# ``pow()``, glibc's for one, agrees with it on these).
_SQUARE_DISAGREEING = (
    0.9156440858938809,
    0.8612114025523432,
    0.8273235598000624,
)


def __test_pow_square_is_a_plain_product__():
    for x in _SQUARE_DISAGREEING:
        assert math.pow(x, 2) == x * x


def __test_pow_half_is_sqrt__():
    for x in _SQUARE_DISAGREEING:
        assert math.pow(x, 0.5) == _math.sqrt(x)


def __test_pow_identity_exponents__():
    for x in _SQUARE_DISAGREEING:
        assert math.pow(x, 1) == x
        assert math.pow(x, 0) == 1.0


def __test_pow_keeps_the_general_path__():
    """Exponents TradingView does NOT shortcut go through the venue's ``pow``.

    MEASURED (CAPITALCOM:BTCUSD@15, ``pow(close / open, 3)``): the venue is not
    ``x * x * x`` on 5537 of 22226 bars.
    """
    for x, expected in ((0.9898623904991166, 0.9698944430229797),
                        (1.002932712766749, 1.0088239659364517),
                        (1.000441389982613, 1.0013247545091832)):
        assert math.pow(x, 3) == expected, x.hex()
        assert x * x * x != expected
    assert math.pow(2, 10) == 1024.0
