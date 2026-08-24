"""
``math.round`` at precision 0 compares against the tie EXACTLY.

The 1e-10 tie tolerance the rounding carries belongs to the decimal scaling of a
non-zero precision, where the exact binary value lands just short of a ``.5``
that the scaled double sits exactly on. Precision 0 scales nothing, so there is
nothing to make up for.

MEASURED on TradingView (BINANCE:BTCUSDT@30): every value below the tie rounds
DOWN however close it sits, while an exact tie still goes away from zero. Both
``math.round(x)`` and ``math.round(x, 0)`` behave this way.

| value                  | short of the tie by | TradingView |
|------------------------|---------------------|-------------|
| 452523.49999999994     | 5.8e-11 (1 ulp)     | 452523      |
| 1048576.4999999998     | 2.3e-10             | 1048576     |
| 8388608.499999998      | 1.9e-09             | 8388608     |
| 100.49999999999999     | 1.4e-14             | 100         |
| 2.4999999999999996     | 4.4e-16             | 2           |
| -452523.49999999994    | 5.8e-11             | -452523     |
| 452523.5000000001      | above the tie       | 452524      |
| 2.5 / -2.5             | on the tie          | 3 / -3      |

The trigger was "Take profit Multi timeframe", whose ``percent()`` helper rounds
``5 / 100 * strategy.position_avg_price / syminfo.mintick``; at an average price
of 90504.7 that is 452523.49999999994, and rounding it up moved every plotted
TP/SL level one tick.
"""
from pynecore.lib import math


_BELOW_THE_TIE = (
    (452523.49999999994, 452523),
    (1048576.4999999998, 1048576),
    (8388608.499999998, 8388608),
    (100.49999999999999, 100),
    (100.49999999999997, 100),
    (2.4999999999999996, 2),
    (-452523.49999999994, -452523),
)


def __test_round_below_the_tie_goes_down__():
    for value, expected in _BELOW_THE_TIE:
        assert math.round(value) == expected
        assert math.round(value, 0) == float(expected)


def __test_round_on_and_above_the_tie__():
    assert math.round(2.5) == 3
    assert math.round(-2.5) == -3
    assert math.round(452523.5) == 452524
    assert math.round(452523.5000000001) == 452524


def __test_scaled_precision_keeps_the_tolerance__():
    """A non-zero precision still forgives the scaling's last-bits shortfall."""
    assert math.round(118152.265, 2) == 118152.27
    assert math.round(100019.7405, 3) == 100019.74
