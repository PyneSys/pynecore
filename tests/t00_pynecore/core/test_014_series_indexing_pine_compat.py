"""
@pyne

Direct unit tests for SeriesImpl.__getitem__ Pine-compat semantics:

- ``series[na]``       -> na      (regression for NA-key TypeError)
- ``series[neg_int]``  -> na      (regression for issue #57: IndexError on negative key)
- ``series[>=size]``   -> na      (pre-existing behavior — must not regress)
- ``series[float]``    -> coerces to int and returns the bar at that offset
"""
from pynecore import lib
from pynecore.core.series import SeriesImpl
from pynecore.types.na import NA


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


def _populate(values: list[float]) -> SeriesImpl:
    """Push values into a fresh SeriesImpl, oldest first. After this call
    ``s[0]`` returns the last pushed value."""
    s = SeriesImpl()
    for i, v in enumerate(values):
        lib.bar_index = i
        s.add(v)
    return s


#
# NA-key indexing — Pine: series[na] -> na
#

def __test_na_key_returns_na__():
    """SeriesImpl[NA] must return NA(T), not raise TypeError"""
    s = _populate([10.0, 20.0, 30.0])
    result = s[NA(int)]
    assert isinstance(result, NA)


def __test_na_float_key_returns_na__():
    """Subscript with an NA[float] key must return na (math.abs(NA[int]) yields NA[float]).

    math.abs(NA[int]) yields NA[float]; subscript with NA[float] must still return na.
    This is the exact case that crashed ict_entry_v2 — ta.highestbars(...) returns na on
    early bars, math.abs propagates as NA[float], and the resulting bar_index[NA[float]]
    used to raise TypeError."""
    s = _populate([10.0, 20.0, 30.0])
    result = s[NA(float)]
    assert isinstance(result, NA)


def __test_na_key_on_empty_series_returns_na__():
    """NA key on an empty series must still return na (no buffer access happens)"""
    s = SeriesImpl()
    result = s[NA(int)]
    assert isinstance(result, NA)


#
# Negative-int indexing — Pine: series[negative] -> na  (issue #57)
#

def __test_negative_int_key_returns_na__():
    """Negative integer index must return NA(T), not raise IndexError"""
    s = _populate([10.0, 20.0, 30.0])
    assert isinstance(s[-1], NA)
    assert isinstance(s[-2], NA)
    assert isinstance(s[-100], NA)


def __test_negative_int_repro_from_issue_57__():
    """Issue #57 pattern ``close[hist_p - j - 1]`` evaluating to a negative integer returns na.

    Exact pattern from issue #57: ``close[hist_p - j - 1]`` where the runtime
    expression evaluates to a negative integer must return na."""
    s = _populate([1.0, 2.0, 3.0, 4.0, 5.0])
    hist_p = 0
    j = 1
    key = hist_p - j - 1  # = -2
    assert isinstance(s[key], NA)


#
# Out-of-range positive — pre-existing behavior, must not regress
#

def __test_positive_key_past_size_returns_na__():
    """Key >= size returns na (history not yet long enough)"""
    s = _populate([10.0, 20.0])
    assert isinstance(s[2], NA)
    assert isinstance(s[100], NA)


#
# Happy path — must not regress
#

def __test_valid_int_key_returns_value__():
    """Plain integer indexing returns the right historical bar"""
    s = _populate([10.0, 20.0, 30.0])
    assert s[0] == 30.0
    assert s[1] == 20.0
    assert s[2] == 10.0


def __test_float_key_coerced_to_int__():
    """Float keys (e.g. from ta.* returning float) coerce to int and index correctly"""
    s = _populate([10.0, 20.0, 30.0])
    assert s[0.0] == 30.0
    assert s[1.7] == 20.0  # int(1.7) == 1
