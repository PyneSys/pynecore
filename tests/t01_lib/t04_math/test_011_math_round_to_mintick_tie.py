"""
Tie handling of ``math.round_to_mintick``.

The tick-grid position is ``number * pricescale / minmove``, and TradingView
decides the tie on the EXACT value of that product, ties going away from zero,
with the same 1e-10 tolerance ``math.round`` carries. Measured on
BINANCE:BTCUSDT@30 (mintick 0.01, minmove 1, pricescale 100) with literal
probes multiplied by a runtime factor so nothing folds at compile time.

The pairs below are what make the law inescapable: every one of these products
is the very same exactly representable ``.5`` in double arithmetic, yet
TradingView splits them by how far the exact product falls short of the tie.
"""
from pynecore.lib import math, syminfo


def setup_function():
    syminfo.mintick = 0.01
    syminfo.minmove = 1
    syminfo.pricescale = 100


def __test_near_tie_within_the_tolerance_rounds_up__():
    """A product short of the tie by less than 1e-10 still rounds up"""
    assert math.round_to_mintick(1.075) == 1.08  # 4.4e-15 short
    assert math.round_to_mintick(1.085) == 1.09
    assert math.round_to_mintick(5243.065) == 5243.07  # 4.0e-11 short
    assert math.round_to_mintick(94130.015) == 94130.02  # 5.8e-11 short
    assert math.round_to_mintick(19999.585) == 19999.59  # 8.7e-11 short


def __test_past_the_tolerance_the_exact_product_decides__():
    """Beyond 1e-10 a printed tie rounds down, however close it looks"""
    assert math.round_to_mintick(16384.245) == 16384.24  # 1.0e-10 short
    assert math.round_to_mintick(94130.155) == 94130.15  # 1.2e-10 short
    assert math.round_to_mintick(94130.045) == 94130.04  # 1.7e-10 short
    assert math.round_to_mintick(94130.025) == 94130.02  # 5.8e-10 short
    # The Ichimoku midpoint that first exposed this: 0.5 * (94500.0 + 93760.93)
    assert math.round_to_mintick(94130.465) == 94130.46


def __test_a_product_above_the_tie_needs_no_tolerance__():
    """On the other side of the tie the double already agrees"""
    assert math.round_to_mintick(100.015) == 100.02
    assert math.round_to_mintick(100.025) == 100.03
    assert math.round_to_mintick(10000.025) == 10000.03
    assert math.round_to_mintick(10000.035) == 10000.04


def __test_negatives_are_symmetric__():
    """Ties go away from zero on both sides, on the same tolerance"""
    assert math.round_to_mintick(-1.075) == -1.08
    assert math.round_to_mintick(-100.025) == -100.03
    assert math.round_to_mintick(-5243.065) == -5243.07
    assert math.round_to_mintick(-16384.245) == -16384.24
    assert math.round_to_mintick(-94130.045) == -94130.04


def __test_off_grid_values_and_zero__():
    """Nothing near a tie takes the exact path, and zero stays zero"""
    assert math.round_to_mintick(0.0) == 0.0
    assert math.round_to_mintick(1.56789918) == 1.57
    assert math.round_to_mintick(-1.56789918) == -1.57
    assert math.round_to_mintick(94130.46) == 94130.46


def __test_a_minmove_other_than_one__():
    """QM1! style grids (mintick 0.025) round to their own tick, not to 0.001"""
    syminfo.mintick = 0.025
    syminfo.minmove = 25
    syminfo.pricescale = 1000
    assert math.round_to_mintick(70.0124) == 70.0
    assert math.round_to_mintick(70.0126) == 70.025
    assert math.round_to_mintick(70.0125) == 70.025
    assert math.round_to_mintick(-70.0125) == -70.025
