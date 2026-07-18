"""
@pyne

Typed na sentinel identity. NA is interned per type (``NA(int) is not
NA(float)``, different hashes), so a function with an int contract must return
the int-typed sentinel: an ``NA(float)`` smuggled through an int-typed return
breaks identity checks, ``.type`` reads and map-key lookups downstream
(regression: ``math.floor``/``ceil`` returned ``cast(int, NA(float))``, the
numeric ``math.max``/``min``/``round`` na branches were hardwired to
``NA(float)``, and the ``array`` reducers lost their element type on
empty/all-na arrays).
"""
from pynecore.lib import math, array
from pynecore.types.na import NA


def main():
    """Dummy main to satisfy the @pyne script loader."""
    pass


#
# math — int contracts return the int-typed sentinel
#

def __test_floor_ceil_int_na__():
    """floor/ceil declare an int return, so their na is NA(int) regardless of input na type"""
    assert math.floor(NA(float)) is NA(int)
    assert math.floor(NA(int)) is NA(int)
    assert math.ceil(NA(float)) is NA(int)
    assert math.ceil(NA(int)) is NA(int)


def __test_max_min_na_follows_operands__():
    """max/min na result carries the operands' numeric contract"""
    assert math.max(NA(int), 3) is NA(int)
    assert math.min(NA(int), 3) is NA(int)
    assert math.max(NA(float), 3) is NA(float)
    assert math.max(NA(int), 3.5) is NA(float)
    assert math.min(1, 2, NA(float)) is NA(float)


def __test_round_na_typed_by_contract__():
    """round without precision is the int overload -> NA(int); with precision -> NA(float)"""
    assert math.round(NA(float)) is NA(int)
    assert math.round(NA(int)) is NA(int)
    assert math.round(NA(float), 2) is NA(float)


#
# array reducers — element type survives the na branches
#

def __test_array_reducers_keep_int_type__():
    """int arrays reduce to ints (Pine returns the element type, not float)"""
    assert array.max([1, 2, 3]) == 3 and isinstance(array.max([1, 2, 3]), int)
    assert array.min([1, 2, 3]) == 1 and isinstance(array.min([1, 2, 3]), int)
    assert array.range([1, 5]) == 4 and isinstance(array.range([1, 5]), int)


def __test_array_reducers_typed_na__():
    """All-na arrays propagate their elements' na type; out-of-range nth keeps the element type"""
    assert array.max([NA(int), NA(int)]) is NA(int)
    assert array.min([NA(float)]) is NA(float)
    assert array.range([NA(int)]) is NA(int)
    assert array.max([1, 2, 3], 5) is NA(int)
    assert array.min([1.0], 5) is NA(float)


def __test_array_reducers_empty_is_typeless_na__():
    """A truly empty array has no element type to honor: the na is the typeless sentinel"""
    for value in (array.max([]), array.min([]), array.range([]), array.mode([])):
        assert isinstance(value, NA)
        assert value.type is None


def __test_array_mode_typed_na__():
    """array.mode propagates the element na type for all-na input and never returns na
    when a real element exists (an NA(int) element must not survive the non-na filter)"""
    assert array.mode([NA(int), NA(int)]) is NA(int)
    assert array.mode([NA(float)]) is NA(float)
    assert array.mode([NA(int), 7, NA(int)]) == 7
