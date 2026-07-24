"""
@pyne
"""
import math

from pynecore.types.na import NA, na_float, na_int, na_str
from pynecore.core.safe_convert import safe_div
from pynecore.lib import map as pine_map
from pynecore.lib import na as is_na
from pynecore.lib import nz


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# Representation: NA(float) is the native nan
#

def __test_na_float_is_native_nan__():
    """NA(float) returns a real IEEE-754 nan float, not an NA object"""
    x = NA(float)
    assert isinstance(x, float)
    assert not isinstance(x, NA)
    assert x != x
    assert math.isnan(x)


def __test_na_float_is_interned__():
    """NA(float) always returns the same interned nan object (identity fast paths)"""
    assert NA(float) is NA(float)
    assert NA(float) is na_float


def __test_non_float_na_stays_object__():
    """int/str/bool na remain interned NA objects — Python int/str have no nan"""
    assert isinstance(NA(int), NA)
    assert isinstance(NA(str), NA)
    assert NA(int) is na_int
    assert NA(str) is na_str


#
# Native IEEE-754 semantics of the float na
#

def __test_nan_arithmetic_propagates__():
    """nan contaminates all arithmetic, like the NA object used to"""
    x = NA(float)
    assert math.isnan(x + 5)
    assert math.isnan(5 - x)
    assert math.isnan(x * 2.0)
    assert math.isnan(x / 3.0)
    assert math.isnan(-x)
    assert math.isnan(abs(x))


def __test_nan_ordering_comparisons_false__():
    """<, >, <=, >=, == on nan are all False (TV-verified P1: same on TV)"""
    x = NA(float)
    assert (x > 0) is False
    assert (x < 0) is False
    assert (x >= 0) is False
    assert (x <= 0) is False
    assert (x == 0) is False
    assert (x == x) is False


def __test_nan_ne_is_true_at_python_level__():
    """nan != x is True in raw IEEE — the ONE operator differing from TV.

    TV evaluates `na != close` as false; compiled Pine `!=` therefore gets a
    compiler-side wrapper (Phase 3). At the Python level the raw IEEE result
    stays — this is what makes the `x != x` na-check idiom work.
    """
    x = NA(float)
    assert (x != x) is True
    assert (x != 0.0) is True


def __test_inf_semantics_are_raw_ieee__():
    """inf participates in arithmetic and comparisons as a normal IEEE value
    (TV-verified P4: inf > 40 is true while na(inf) is also true)"""
    inf = math.inf
    assert (inf > 40) is True
    assert (-inf < 0) is True
    assert inf + 5 == inf
    assert math.isnan(inf - inf)
    assert math.isnan(inf * 0)


#
# NA object semantics unchanged for non-float types
#

def __test_na_object_comparisons_always_false__():
    """NA object comparisons still return False, including !="""
    x = na_int
    assert (x > 40) is False
    assert (x < 40) is False
    assert (x == 40) is False
    assert (x != 40) is False
    assert (x != x) is False


def __test_na_object_arithmetic_propagates_self__():
    """NA object arithmetic returns the same NA object"""
    x = na_int
    assert (x + 5) is x
    assert (5 + x) is x
    assert (x * 5) is x
    assert (x - 5) is x
    assert abs(x) is x
    assert (-x) is x


#
# safe_div — raw IEEE results (P4)
#

def __test_safe_div_normal_division__():
    """Normal division returns the float result"""
    assert safe_div(10.0, 2.0) == 5.0
    assert safe_div(1.0, 4.0) == 0.25


def __test_safe_div_positive_by_zero_is_inf__():
    """positive / 0 returns raw +inf"""
    assert safe_div(2155.0, 0.0) == math.inf
    assert safe_div(1.0, 0.0) == math.inf


def __test_safe_div_negative_by_zero_is_neg_inf__():
    """negative / 0 returns raw -inf"""
    assert safe_div(-2155.0, 0.0) == -math.inf
    assert safe_div(-1.0, 0.0) == -math.inf


def __test_safe_div_zero_by_zero_is_nan__():
    """0 / 0 returns nan"""
    assert math.isnan(safe_div(0.0, 0.0))


def __test_safe_div_na_input_returns_nan__():
    """na input propagates as nan"""
    assert math.isnan(safe_div(NA(float), 2.0))
    assert math.isnan(safe_div(2.0, NA(float)))
    result = safe_div(na_int, 2.0)
    assert math.isnan(result)


def __test_safe_div_result_pine_semantics__():
    """Integration-style: Gekko Machine dropPercent > 40 pattern"""
    profit_drop = 2155.0
    max_profit = 0.0
    drop_percent = safe_div(profit_drop, max_profit) * 100
    # inf * 100 = inf
    assert drop_percent == math.inf
    # Pine semantic: inf > 40 is True
    assert (drop_percent > 40) is True
    assert (drop_percent > 20) is True


#
# is_na — the na() predicate is not-isfinite for floats (P4)
#

def __test_is_na_on_non_finite_floats__():
    """na() reports inf/-inf/nan as na (TV-verified: na(inf) is true)"""
    assert is_na(float('inf')) is True
    assert is_na(float('-inf')) is True
    assert is_na(float('nan')) is True
    assert is_na(NA(float)) is True


def __test_is_na_on_na_objects__():
    """NA objects still report as na"""
    assert is_na(na_int) is True
    assert is_na(na_str) is True


def __test_is_na_on_finite_values__():
    """Finite values are not na"""
    assert is_na(0.0) is False
    assert is_na(42) is False
    assert is_na(-1.5) is False


#
# nz — replacement follows the na() predicate (inf counts as na, P4)
#

def __test_nz_replaces_nan__():
    assert nz(NA(float), -5.0) == -5.0
    assert nz(float('nan')) == 0


def __test_nz_replaces_inf__():
    """TV-verified P4: nz(inf, -5) is -5"""
    assert nz(float('inf'), -5.0) == -5.0
    assert nz(float('-inf'), -5.0) == -5.0


def __test_nz_keeps_finite_and_na_objects_replaced__():
    assert nz(3.25, -5.0) == 3.25
    assert nz(0.0, -5.0) == 0.0
    assert nz(na_int, 7) == 7


#
# map — an na key is storable and retrievable (TV-verified P5)
#

def __test_map_nan_key_roundtrip__():
    """map.put with a float na key, then get/contains with a DIFFERENT nan object"""
    m = pine_map.new()
    pine_map.put(m, NA(float), 1.0)
    assert pine_map.size(m) == 1
    assert pine_map.contains(m, float('nan')) is True
    assert pine_map.get(m, float('nan')) == 1.0


def __test_map_nan_key_overwrite_and_remove__():
    """Two nan keys canonicalize to one entry; remove finds it too"""
    m = pine_map.new()
    pine_map.put(m, float('nan'), 1.0)
    old = pine_map.put(m, math.nan, 2.0)
    assert old == 1.0
    assert pine_map.size(m) == 1
    assert pine_map.remove(m, float('nan')) == 2.0
    assert pine_map.size(m) == 0


def __test_map_normal_keys_unaffected__():
    m = pine_map.new()
    pine_map.put(m, 1.5, 'a')
    pine_map.put(m, float('nan'), 'b')
    assert pine_map.get(m, 1.5) == 'a'
    assert pine_map.size(m) == 2


#
# ``in`` must not iterate na forever (__getitem__ returns self unboundedly)
#

def __test_in_operator_on_na_is_false_not_infinite__():
    """``x in na`` returns False immediately via __contains__.

    Without __contains__ the ``in`` operator falls back to the sequence
    protocol, and NA.__getitem__ (returns self for every index, never raising
    IndexError) makes it loop forever.
    """
    assert ('anything' in NA(str)) is False
    assert (42 in na_int) is False
    assert (None in NA(bool)) is False
