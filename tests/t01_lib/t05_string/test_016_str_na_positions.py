"""
An ``na`` position is a concrete position for ``str.substring``/``str.replace``.

Pine's ``int`` truncation at the consuming slot (see
``test_015_str_int_typed_positions``) reached those slots through a bare
``int()``, and ``NA.__int__`` raises -- so from PyneCore v6.9.0 every one of
these calls died with ``TypeError: NA cannot be converted to int`` where
TradingView answers a string.

MEASURED on TradingView (BITSTAMP:BTCUSD@60, logged with ``log.info``):

| expression                          | TradingView |
|-------------------------------------|-------------|
| `str.substring("hello", na)`        | `hello`     |
| `str.substring("hello", 3, na)`     | `lo`        |
| `str.substring("hello", 1, na)`     | `ello`      |
| `str.substring("hello", na, na)`    | `hello`     |
| `str.replace("aaa", "a", "-", na)`  | `-aa`       |
| `str.replace("abc", "", "-", na)`   | `-abc`      |

So an na START is 0, while an na END reaches the end of the source the way an
omitted one does -- the two positions answer na differently. An explicit end
below the begin is rejected at compile time (CE10276), so the na end can never
be read as 0.

Both na representations are exercised: bare ``na`` in a script is the typeless
``NA`` object (``NA(None)``, ``lib._na_none``), whose ``__int__`` raises
``TypeError``, while an int-typed na travels as a native ``nan``
(``NA(int)`` is interned to one), whose ``int()`` raises ``ValueError``. Only the first is what
issue #79 hit, and only ``NA`` answers ``False`` to ``!=`` as well as ``==``.
"""
import pytest

from pynecore.lib import string
from pynecore.types.na import NA

# Bare ``na`` in a script, and an int-typed na, which is a native nan
NAS = (NA(None), NA(int))


@pytest.mark.parametrize('na', NAS)
def __test_substring_na_begin_is_zero__(na):
    """An na begin position starts at the beginning of the source"""
    assert string.substring("hello", na) == "hello"
    assert string.substring("hello", na) == string.substring("hello", 0)


@pytest.mark.parametrize('na', NAS)
def __test_substring_na_end_reaches_the_source_end__(na):
    """An na end position is the end of the source, not position 0"""
    assert string.substring("hello", 3, na) == "lo"
    assert string.substring("hello", 1, na) == "ello"
    # ... which is exactly what omitting it does
    assert string.substring("hello", 3, na) == string.substring("hello", 3)


@pytest.mark.parametrize('na', NAS)
def __test_substring_both_positions_na__(na):
    """Both positions na yields the whole source"""
    assert string.substring("hello", na, na) == "hello"


@pytest.mark.parametrize('na', NAS)
def __test_replace_na_occurrence_is_the_first__(na):
    """An na occurrence replaces the first match"""
    assert string.replace("aaa", "a", "-", na) == "-aa"
    assert string.replace("aaa", "a", "-", na) == string.replace("aaa", "a", "-", 0)
    # An empty target is an insertion point, so an na occurrence inserts at the front
    assert string.replace("abc", "", "-", na) == "-abc"


@pytest.mark.parametrize('na', NAS)
def __test_na_source_propagates_instead_of_crashing__(na):
    """An na source yields na at every position combination

    The end position defaults to ``len(source)``, which an na source cannot
    answer, so the source has to be resolved before any position is.
    """
    for end in (None, 0, 1, na):
        result = string.substring(NA(str), 0) if end is None else string.substring(NA(str), 0, end)
        assert isinstance(result, NA), f"end={end!r} gave {result!r}"


@pytest.mark.parametrize('na', NAS)
def __test_repeat_na_count_yields_na__(na):
    """A typeless na repeat count is na, not a TypeError"""
    assert isinstance(string.repeat("a", NA(None)), NA)
    assert isinstance(string.repeat("a", na), NA)


@pytest.mark.parametrize('na', NAS)
def __test_repeat_na_count_is_na__(na):
    """An na repeat count answers na instead of raising"""
    # The guard read ``repeat != repeat``, which is False for the NA object --
    # only the nan spelling ever reached it.
    assert isinstance(string.repeat("ab", na), NA)
