"""
``math.pow`` takes TradingView's exact shortcuts for four exponents.

MEASURED on TradingView (BINANCE:BTCUSDT@30, 8000 bars, base ``x`` in
[0.3, 1.3] built from ``ta.correlation``): ``pow(x, 2) - x * x``,
``pow(x, 0.5) - sqrt(x)``, ``pow(x, 1) - x`` and ``pow(x, 0) - 1`` were zero on
EVERY bar, while ``pow(x, -1) - 1 / x`` was non-zero on 3 and
``pow(x, 3) - x * x * x`` on 2149 -- so only these four are shortcuts.

The platform ``pow()`` behind Python's ``**`` disagrees with the shortcut on a
handful of those bases, which a recursive script turns into a visible plot
divergence, so the bases below are exactly the ones measured to disagree.
"""
import math as _math

from pynecore.lib import math


# Bases where CPython's ``**`` differs from the shortcut on this platform.
_SQUARE_DISAGREEING = (
    0.9156440858938809,
    0.8612114025523432,
    0.8273235598000624,
)


def __test_pow_square_is_a_plain_product__():
    for x in _SQUARE_DISAGREEING:
        assert math.pow(x, 2) == x * x

    # The shortcut is what differs from ``**`` -- otherwise the test proves
    # nothing about the code under test.
    assert any(x ** 2 != x * x for x in _SQUARE_DISAGREEING)


def __test_pow_half_is_sqrt__():
    for x in _SQUARE_DISAGREEING:
        assert math.pow(x, 0.5) == _math.sqrt(x)


def __test_pow_identity_exponents__():
    for x in _SQUARE_DISAGREEING:
        assert math.pow(x, 1) == x
        assert math.pow(x, 0) == 1.0


def __test_pow_keeps_the_general_path__():
    """Exponents TradingView does NOT shortcut stay on the platform ``pow()``."""
    x = 1.2718281828459045
    assert math.pow(x, 3) == x ** 3
    assert math.pow(x, -1) == x ** -1
    assert math.pow(2, 10) == 1024.0
