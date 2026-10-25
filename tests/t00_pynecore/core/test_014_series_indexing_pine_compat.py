"""
@pyne

Direct unit tests for SeriesImpl.__getitem__ Pine-compat semantics:

- ``series[na]``       -> series[0]  (TradingView treats an na offset as the current bar)
- ``series[neg_int]``  -> na      (regression for issue #57: IndexError on negative key)
- ``series[>=size]``   -> na      (pre-existing behavior — must not regress)
- ``series[float]``    -> coerces to int and returns the bar at that offset
"""
from pynecore import lib
from pynecore.core.series import SeriesImpl
from pynecore.lib import na
from pynecore.types.na import NA, na_bool, na_float, set_bool_na


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
# NA-key indexing — Pine: series[na] -> series[0] (na offset is the current bar)
#

def __test_na_key_returns_current_bar__():
    """SeriesImpl[NA] returns the current bar (offset 0), matching TradingView.

    Verified against TradingView: ``high[na]`` returns ``high[0]`` (the current
    bar's value), not na. The subscript must also not raise TypeError on an NA key."""
    s = _populate([10.0, 20.0, 30.0])
    assert s[NA(int)] == 30.0


def __test_na_float_key_returns_current_bar__():
    """Subscript with an NA[float] key returns the current bar (offset 0).

    math.abs(NA[int]) yields NA[float]; subscript with NA[float] must behave like
    any na offset and return series[0]. This is the exact case from ict_entry_v2 —
    ta.highestbars(...) returns na on early bars, math.abs propagates as NA[float],
    and the resulting high[NA[float]] must resolve to the current bar (it used to
    raise TypeError, was then wrongly made na, and TradingView in fact returns offset 0)."""
    s = _populate([10.0, 20.0, 30.0])
    assert s[NA(float)] == 30.0


def __test_na_key_on_empty_series_returns_na__():
    """NA key on an empty series must still return na (no buffer access happens)"""
    s = SeriesImpl()
    result = s[NA(int)]
    assert na(result)


#
# Negative-int indexing — Pine: series[negative] -> na  (issue #57)
#

def __test_negative_int_key_returns_na__():
    """Negative integer index must return NA(T), not raise IndexError"""
    s = _populate([10.0, 20.0, 30.0])
    assert na(s[-1])
    assert na(s[-2])
    assert na(s[-100])


def __test_negative_int_repro_from_issue_57__():
    """Issue #57 pattern ``close[hist_p - j - 1]`` evaluating to a negative integer returns na.

    Exact pattern from issue #57: ``close[hist_p - j - 1]`` where the runtime
    expression evaluates to a negative integer must return na."""
    s = _populate([1.0, 2.0, 3.0, 4.0, 5.0])
    hist_p = 0
    j = 1
    key = hist_p - j - 1  # = -2
    assert na(s[key])


#
# Out-of-range positive — pre-existing behavior, must not regress
#

def __test_positive_key_past_size_returns_na__():
    """Key >= size returns na (history not yet long enough)"""
    s = _populate([10.0, 20.0])
    assert na(s[2])
    assert na(s[100])


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


#
# Out-of-range na of a series built without an element type — the na of what it stores
#

def __test_untyped_series_reads_the_na_of_what_it_stores__():
    """A series built without an element type (an ``inline_series`` buffer) reads the
    na of its stored value's type out of range, as a ``Series[<type>]`` does: the nan
    for a number, the empty string for a string, ``NA(<class>)`` for an object."""
    class Pivot:
        pass

    for value, expected in ((1.5, na_float), (3, na_float), ('abc', ''),
                            (Pivot(), NA(Pivot)), ([1.0], NA(list)), (NA(Pivot), NA(Pivot))):
        s = _populate([value])
        assert s[1] is expected, (value, s[1])
        assert s[-1] is expected, (value, s[-1])


def __test_untyped_series_keeps_the_bool_na_of_the_script_mode__():
    """A stored bool makes the out-of-range read the bool na the script runs with:
    false under the two-state bool, the interned na under the three-state one."""
    try:
        set_bool_na(False)
        assert _populate([True])[1] is False
        set_bool_na(True)
        assert _populate([True])[1] is na_bool
    finally:
        set_bool_na(False)


def __test_untyped_series_learns_its_na_from_the_newest_typed_value__():
    """A typeless na stored first says nothing about the type: the read stays the typeless
    na until a typed value is stored, and the na is kept once found."""
    s = _populate([NA(None)])
    assert isinstance(s[1], NA)
    lib.bar_index = 1
    s.add(2.0)
    assert s[2] is na_float
    lib.bar_index = 2
    s.add(NA(None))
    assert s[3] is na_float
