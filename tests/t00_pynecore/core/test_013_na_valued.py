"""
@pyne
"""
import math

from pynecore.types.na import NA, na_inf, na_neg_inf, na_nan, na_float
from pynecore.core.safe_convert import safe_div
from pynecore.lib import na as is_na


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# Identity and isinstance
#

def __test_valued_na_are_singletons__():
    """Valued NAs are cached singletons — repeated construction returns the same instance"""
    from pynecore.types.na import _InfType, _NegInfType, _NanType
    assert NA(_InfType) is na_inf
    assert NA(_NegInfType) is na_neg_inf
    assert NA(_NanType) is na_nan


def __test_valued_na_are_instances_of_NA__():
    """isinstance(x, NA) stays True for valued singletons — lib-internal NA-skip guards must keep working"""
    assert isinstance(na_inf, NA)
    assert isinstance(na_neg_inf, NA)
    assert isinstance(na_nan, NA)


def __test_valued_na_distinct_from_plain_na__():
    """Valued singletons must not collide with plain NA(float)"""
    assert na_inf is not na_float
    assert na_neg_inf is not na_float
    assert na_nan is not na_float


#
# Comparisons — Pine IEEE-754 semantics for valued NA
#

def __test_na_inf_greater_than_finite__():
    """na_inf behaves like +inf in comparisons: inf > any_finite is True"""
    assert (na_inf > 40) is True
    assert (na_inf > 0) is True
    assert (na_inf > -1e300) is True
    assert (na_inf >= 1e300) is True


def __test_na_inf_not_less_than_finite__():
    """inf < any_finite is False"""
    assert (na_inf < 40) is False
    assert (na_inf <= 40) is False


def __test_na_neg_inf_less_than_finite__():
    """-inf < any finite is True (IEEE-754: -inf is the smallest)"""
    assert (na_neg_inf < 0) is True
    assert (na_neg_inf < -1e300) is True
    assert (na_neg_inf > -1e300) is False


def __test_na_nan_comparisons_always_false__():
    """NaN comparisons always return False (IEEE-754)"""
    assert (na_nan > 0) is False
    assert (na_nan < 0) is False
    assert (na_nan == 0) is False
    assert (na_nan >= 0) is False
    assert (na_nan <= 0) is False


def __test_plain_na_comparisons_always_false__():
    """Plain NA comparisons still return False — backwards compatible"""
    plain = na_float
    assert (plain > 40) is False
    assert (plain < 40) is False
    assert (plain == 40) is False
    assert (plain >= 40) is False
    assert (plain <= 40) is False


def __test_valued_na_against_each_other__():
    """Comparisons between valued NAs follow IEEE-754"""
    assert (na_inf > na_neg_inf) is True
    assert (na_neg_inf < na_inf) is True
    # inf == inf is True in IEEE-754
    assert (na_inf == na_inf) is True
    # nan vs anything is False
    assert (na_nan == na_nan) is False
    assert (na_inf > na_nan) is False
    assert (na_neg_inf < na_nan) is False


#
# Arithmetic — normalization back to singletons
#

def __test_inf_times_positive__():
    """inf * positive finite is still inf"""
    assert na_inf * 100 is na_inf
    assert 100 * na_inf is na_inf
    assert na_inf * 1.5 is na_inf


def __test_inf_times_negative__():
    """inf * negative finite is -inf"""
    assert na_inf * -1 is na_neg_inf
    assert -1 * na_inf is na_neg_inf


def __test_inf_times_zero_is_nan__():
    """inf * 0 is nan (IEEE-754)"""
    assert na_inf * 0 is na_nan
    assert 0 * na_inf is na_nan


def __test_inf_minus_inf_is_nan__():
    """inf - inf is nan"""
    assert (na_inf - na_inf) is na_nan
    assert (na_inf + na_neg_inf) is na_nan


def __test_neg_inf_plus_finite__():
    """-inf + finite stays -inf"""
    assert (na_neg_inf + 5) is na_neg_inf
    assert (5 + na_neg_inf) is na_neg_inf


def __test_negation__():
    """-na_inf == na_neg_inf, -na_neg_inf == na_inf"""
    assert -na_inf is na_neg_inf
    assert -na_neg_inf is na_inf
    # -nan stays nan
    assert -na_nan is na_nan


def __test_abs__():
    """abs(-inf) == inf, abs(nan) stays nan"""
    assert abs(na_neg_inf) is na_inf
    assert abs(na_inf) is na_inf
    assert abs(na_nan) is na_nan


def __test_nan_propagation__():
    """nan contaminates all arithmetic"""
    assert (na_nan + 5) is na_nan
    assert (na_nan * 5) is na_nan
    assert (na_nan - 5) is na_nan
    assert (5 + na_nan) is na_nan


def __test_plain_na_arithmetic_unchanged__():
    """Plain NA arithmetic still returns plain NA — lib code stays functional"""
    plain = na_float
    assert (plain + 5) is plain
    assert (plain * 5) is plain
    assert (plain - 5) is plain
    assert (5 + plain) is plain


#
# safe_div — Pine-compatible division
#

def __test_safe_div_normal_division__():
    """Normal division returns the float result"""
    assert safe_div(10.0, 2.0) == 5.0
    assert safe_div(1.0, 4.0) == 0.25


def __test_safe_div_positive_by_zero_is_inf__():
    """positive / 0 returns na_inf singleton"""
    assert safe_div(2155.0, 0.0) is na_inf
    assert safe_div(1.0, 0.0) is na_inf


def __test_safe_div_negative_by_zero_is_neg_inf__():
    """negative / 0 returns na_neg_inf singleton"""
    assert safe_div(-2155.0, 0.0) is na_neg_inf
    assert safe_div(-1.0, 0.0) is na_neg_inf


def __test_safe_div_zero_by_zero_is_nan__():
    """0 / 0 returns na_nan singleton"""
    assert safe_div(0.0, 0.0) is na_nan


def __test_safe_div_na_input_returns_plain_na__():
    """NA input propagates as plain NA, not a valued singleton"""
    result = safe_div(na_float, 2.0)
    assert isinstance(result, NA)
    assert result is not na_inf
    assert result is not na_nan


def __test_safe_div_result_pine_semantics__():
    """Integration-style: Gekko Machine dropPercent > 40 pattern"""
    profit_drop = 2155.0
    max_profit = 0.0
    drop_percent = safe_div(profit_drop, max_profit) * 100
    # inf * 100 = inf
    assert drop_percent is na_inf
    # Pine semantic: inf > 40 is True
    assert (drop_percent > 40) is True
    assert (drop_percent > 20) is True


#
# is_na — predicate must recognize valued NAs as NA
#

def __test_is_na_on_valued_singletons__():
    """Pine's na() predicate must return True for valued NA singletons"""
    assert is_na(na_inf) is True
    assert is_na(na_neg_inf) is True
    assert is_na(na_nan) is True


def __test_is_na_on_plain_na__():
    """Plain NA still returns True"""
    assert is_na(na_float) is True


def __test_is_na_on_native_float_inf_nan__():
    """Defensive: native float inf/nan should also be recognized as na"""
    assert is_na(float('inf')) is True
    assert is_na(float('-inf')) is True
    assert is_na(float('nan')) is True


def __test_is_na_on_finite_values__():
    """Finite values are not na"""
    assert is_na(0.0) is False
    assert is_na(42) is False
    assert is_na(-1.5) is False


#
# ta.*-style internal guard: isinstance(mfv, NA) still filters out valued NA
#

def __test_isinstance_na_guards_still_work__():
    """Simulates ta.accdist pattern: valued-NA operand must be caught by isinstance check"""
    ad = 100.0
    # Mimic mfm = 0/0 -> na_nan; mfv = na_nan * volume -> na_nan
    mfm = safe_div(0.0, 0.0)
    mfv = mfm * 1000  # volume
    assert mfv is na_nan
    assert isinstance(mfv, NA)
    # The internal guard blocks the accumulation
    if not isinstance(mfv, NA):
        ad += mfv  # would poison ad
    assert ad == 100.0  # unchanged, correct