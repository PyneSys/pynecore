"""Pine's explicit ``bool()`` cast converts floats tolerantly.

The reference values come from a TradingView probe run on BINANCE:BTCUSDT 30m,
with every constant multiplied by a runtime factor so nothing is folded at
compile time: ``bool(1e-10)`` is false and ``bool(1.000001e-10)`` is true at
both signs, ``bool(2e-15)`` and ``bool(na)`` are false, and every one of the
28149 bars agrees. It is the same threshold the implicit contexts use, so the
cast and the truthiness rewrite must not disagree.
"""
import math

from pynecore.core.pine_cast import cast_bool
from pynecore.core.pine_compare import EPSILON
from pynecore.types.na import NA


def __test_float_within_the_tolerance_is_false__():
    """ A float no farther than EPSILON from zero casts to False """
    assert cast_bool(0.0) is False
    assert cast_bool(-0.0) is False
    assert cast_bool(2.3684757858670005e-15) is False
    assert cast_bool(-4.5e-14) is False
    assert cast_bool(9.9e-11) is False


def __test_boundary_belongs_to_false__():
    """ Exactly the threshold is still False; one ulp above is True """
    assert cast_bool(EPSILON) is False
    assert cast_bool(-EPSILON) is False
    assert cast_bool(math.nextafter(EPSILON, 1.0)) is True
    assert cast_bool(-math.nextafter(EPSILON, 1.0)) is True
    assert cast_bool(1.1e-10) is True
    assert cast_bool(1e-9) is True


def __test_na_is_false__():
    """ Both na representations cast to False """
    assert cast_bool(float('nan')) is False
    assert cast_bool(NA(float)) is False
    assert cast_bool(NA(bool)) is False


def __test_non_floats_keep_exact_semantics__():
    """ Ints and bools are not put on the tolerance grid """
    assert cast_bool(0) is False
    assert cast_bool(1) is True
    assert cast_bool(-1) is True
    assert cast_bool(False) is False
    assert cast_bool(True) is True


def __test_infinities_are_true__():
    """ An infinity is outside every band, at both signs """
    assert cast_bool(float('inf')) is True
    assert cast_bool(float('-inf')) is True
