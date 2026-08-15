"""
Tie handling of ``math.round(number, precision)``.

TradingView rounds the exact binary value of the double scaled by ``10 **
precision``, ties going away from zero, and accepts anything within 1e-10 of the
tie as a tie. Measured on BINANCE:BTCUSDT@30 (probe ``roundprobe``, 28397 bars of
``close * 1.21``, ``close * 0.79`` and ``close / 3`` at precision 2, 3 and 5) plus
literal probes for the boundary cases below.
"""
from pynecore.lib import math


def __test_ties_round_away_from_zero__():
    """A tie in the shortest decimal form rounds away from zero, not half-even"""
    assert math.round(2.5) == 3
    assert math.round(-2.5) == -3
    # 2.675 is 2.67499999999999982 as a double: 1.8e-14 short of the scaled tie,
    # well inside the tolerance, so it rounds up like the decimal it prints as.
    assert math.round(2.675, 2) == 2.68
    assert math.round(-2.675, 2) == -2.68


def __test_tolerance_admits_a_near_tie__():
    """A value just short of the tie still rounds up while it is within 1e-10"""
    # 118152.265 * 100 falls 5.8e-11 short of 11815226.5 in exact arithmetic.
    assert math.round(118152.265, 2) == 118152.27
    # 50412.1515 * 1000 falls 5.8e-11 short of 50412151.5.
    assert math.round(50412.1515, 3) == 50412.152


def __test_tolerance_stops_at_1e_10__():
    """Past the tolerance the exact binary value decides, even on a printed tie"""
    # Both scale to an exactly representable .5 in double arithmetic, so only the
    # exact value separates them: 1.2e-10 short of the tie versus 5.8e-11 above.
    assert math.round(100019.7405, 3) == 100019.740
    assert math.round(105826.5055, 3) == 105826.505
    # The stop-loss level that first exposed this: 101470.95 * 1.21 prints as
    # 122779.8495 but is 3.4e-9 short of the scaled tie.
    assert math.round(122779.8495, 3) == 122779.849


def __test_value_above_the_tie_always_rounds_up__():
    """No tolerance is needed on the other side of the tie"""
    assert math.round(40000.0005, 3) == 40000.001
    assert math.round(99999.9995, 3) == 100000.0


def __test_precision_beyond_the_double_grid_is_the_identity__():
    """A precision no double can express leaves the number alone"""
    assert math.round(1e30, 2) == 1e30
    assert math.round(0.0, 3) == 0.0
