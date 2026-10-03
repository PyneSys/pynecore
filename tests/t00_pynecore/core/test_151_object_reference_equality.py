"""
Pine objects compare by reference: an array search finds the very object it was given,
never a distinct object with equal fields, and the answer does not depend on the Python
version running the script.
"""
from pynecore.core.pine_udt import udt, udt_copy
from pynecore.lib import array, chart
from pynecore.types.footprint import Footprint
from pynecore.types.na import NA
from pynecore.types.volume_row import VolumeRow

# Expected values measured on TradingView (probes udt_eq / cp_eq, 2026-10-03)

na_float = NA(float)


@udt
class _Point:
    x: float = na_float
    y: int = 1


def __test_udt_search_matches_the_same_object_only__():
    """ A distinct UDT with equal fields is not found """
    c = _Point(1.0, 1)
    d = _Point(1.0, 1)
    items = [c]
    assert array.includes(items, c)
    assert not array.includes(items, d)
    assert array.indexof(items, c) == 0
    assert array.indexof(items, d) == -1
    assert array.lastindexof(items, d) == -1


def __test_udt_copy_is_a_distinct_object__():
    """ A copy is not found where the original is """
    c = _Point(1.0, 1)
    assert not array.includes([c], udt_copy(c))


def __test_udt_with_na_fields_compares_by_reference__():
    """
    Two UDTs sharing one ``na`` field value are distinct objects. A field-wise dataclass
    ``__eq__`` called them equal on CPython 3.11/3.12 and unequal on 3.13+.
    """
    a = _Point(na_float, 1)
    b = _Point(na_float, 1)
    assert a == a
    assert a != b
    assert array.includes([a], a)
    assert not array.includes([a], b)


def __test_chart_point_search_matches_the_same_object_only__():
    """ A distinct ``chart.point`` with equal coordinates is not found """
    c = chart.point.from_index(5, 1.0)
    d = chart.point.from_index(5, 1.0)
    items = [c]
    assert array.includes(items, c)
    assert not array.includes(items, d)
    assert not array.includes(items, chart.point.copy(c))
    assert array.indexof(items, d) == -1


def __test_footprint_objects_are_distinct__():
    """ Field-less footprint objects are still distinct references """
    assert Footprint() != Footprint()
    assert VolumeRow() != VolumeRow()
