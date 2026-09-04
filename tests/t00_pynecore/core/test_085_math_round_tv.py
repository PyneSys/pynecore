"""
``math.round(number, precision)`` reproduces TradingView bit for bit.

TradingView keeps the integer part of the number and rounds only its fraction:
``i + round(f * scale) / scale`` with ``scale = 1 / 10**-precision``, ties going
away from zero, decided on the EXACT product of the two doubles with a 1e-10
tolerance. No precision, precision 0 and any negative precision are the same
tolerance-free integer rounding; a precision above 16 acts as 16.

Every expected value below is a TradingView measurement (FX:EURUSD@60, Pine v6,
probes mr1-mr14 and mrp1-mrp3; the doubles were recovered exactly by logging the
result times 2**30).
"""
from pynecore.lib import math
from pynecore.types.na import NA


def __test_fractional_precision_scales_only_the_fraction__():
    """2.34567 @ 0.25 is 2 + 10**-0.25, nowhere near the input"""
    assert math.round(2.34567, 0.25) == 2.5623413251903493
    assert math.round(2.34567, 1.75) == 2.3378730879073952
    assert math.round(1234.5678, 1.75) == 1234.5690494112125
    assert math.round(0.5, 1.75) == 0.4979182348108984
    assert math.round(1.05, 2.5) == 1.0505964425626941
    assert math.round(-1.05, 1.5) == -1.0632455532033676


def __test_scale_is_the_reciprocal_of_the_negative_power__():
    """1 / 10**-p differs from 10**p by an ulp at 0.75, 1.75, 2.5 and 3.5"""
    assert math.round(2.34567, 0.75) == 2.3556558820077846
    assert math.round(0.5, 0.75) == 0.5334838230116768
    assert math.round(2.34567, 2.5) == 2.3446882649583536
    assert math.round(2.34567, 3.5) == 2.345636948256404
    assert math.round(2.34567, 0.5) == 2.316227766016838


def __test_integer_part_is_added_back_unchanged__():
    """The result is i + n / scale, not the nearest double to the decimal"""
    assert math.round(1.56789, 2) == 1.5699999999999998
    assert math.round(0.0625, 3) == 0.063
    assert math.round(-0.0625, 3) == -0.063


def __test_ties_go_away_from_zero__():
    assert math.round(2.25, 1) == 2.3
    assert math.round(-2.25, 1) == -2.3
    assert math.round(2.675, 2) == 2.68
    assert math.round(-2.675, 2) == -2.68
    assert math.round(2.5) == 3.0
    assert math.round(-2.5) == -3.0


def __test_tie_tolerance_is_1e_10_on_the_exact_product__():
    """13.887175 and 63.986275 scale to the same double at precision 5"""
    # Exact products 8.4e-11 and 8.8e-11 short of the tie: both inside 1e-10
    # 13.887175: the exact product is 9.7e-11 short of the tie, inside 1e-10.
    # 63.986275: 1.02e-10 short (the scale is the double 99999.99999999999, and
    # the product with it, not with 10**5, decides), so it rounds down.
    assert math.round(13.887175, 5) == 13.88718
    assert math.round(63.986275, 5) == 63.98627
    assert math.round(18.496225, 5) == 18.49622
    # The threshold itself: 3.16e-11 short rounds up, 1.0e-10 short rounds down
    assert math.round(0.149999999996838, 1) == 0.2
    assert math.round(0.14999999999, 1) == 0.1
    assert math.round(1.15, 1) == 1.2


def __test_integer_rounding_has_no_tolerance__():
    """Precision 0 and the one-argument form compare against the tie exactly"""
    assert math.round(0.4999999999999) == 0.0
    assert math.round(0.4999999999999, 0) == 0.0
    assert math.round(1000.4999999999, 0) == 1000.0


def __test_negative_precision_is_integer_rounding__():
    """A negative precision is clamped to 0, it does not round to tens"""
    assert math.round(1234.5678, -0.5) == 1235.0
    assert math.round(1234.5678, -1.5) == 1235.0
    assert math.round(2.5, -3) == 3.0
    assert math.round(-2.5, -2.5) == -3.0
    assert math.round(2.34567, -3) == 2.0


def __test_precision_above_16_acts_as_16__():
    assert math.round(1.95385001950279, 16) == 1.9538500195027901
    assert math.round(1.95385001950279, 17) == 1.9538500195027901
    assert math.round(1.95385001950279, 100) == 1.9538500195027901
    # Where 10**-precision underflows the number comes back unchanged
    assert math.round(2.34567, 309) == 2.34567
    assert math.round(1234.5678, 400) == 1234.5678


def __test_integers_beyond_2_pow_52_are_left_alone__():
    """Every double is an integer there; the +0.5 trick would round to even"""
    assert math.round(2.0 ** 52 + 1) == 2.0 ** 52 + 1
    assert math.round(-(2.0 ** 52 + 1), 0) == -(2.0 ** 52 + 1)
    assert math.round(2.0 ** 53 - 1) == 2.0 ** 53 - 1
    assert math.round(2.0 ** 51 + 0.5) == 2.0 ** 51 + 1
    assert math.round(2.0 ** 51 + 0.5, 2) == 2.0 ** 51 + 0.5


def __test_one_argument_result_is_an_integral_float__():
    result = math.round(2.34567)
    assert result == 2.0 and isinstance(result, float)


def __test_na_handling__():
    # The one-argument form keeps its int contract, the precision form is a float
    assert math.round(NA(float)) is NA(int)
    assert math.round(NA(float), 2) is NA(float)
    # An na precision is the same "not given": the one-argument form
    assert math.round(2.5, NA(int)) == 3.0
    assert math.round(2.34567, NA(int)) == 2.0
