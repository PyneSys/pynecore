"""
A field of an na object stops the script.

Measured on TradingView (CAPITALCOM:EURUSD 1D): reading or writing a field of an
na user-defined object or ``chart.point`` is a runtime error -- directly, through a
method, nested (``o.inner.v``), inside ``na(t.v)``, on an ``array.new<T>(1)``
element, on an object history before its first bar, and when the object is
copied. A method called on the na itself runs (``this`` is na), and an na
drawing keeps answering na through its getters.
"""
import pytest

from pynecore.core.pine_udt import udt, udt_copy
from pynecore.lib import chart
from pynecore.types.chart import ChartPoint
from pynecore.types.line import Line
from pynecore.types.na import NA, NaFieldError


@udt
class Inner:
    v: float = 1.0


@udt
class Outer:
    inner: Inner = NA(Inner)


@udt
class Kind:
    type: float = 1.0
    v: float = 1.0


def __test_reading_a_field_of_an_na_object_raises__():
    """A field read on an na user-defined object or chart point is an error"""
    with pytest.raises(NaFieldError, match="field 'v'"):
        _ = NA(Inner).v
    with pytest.raises(NaFieldError, match="field 'price'"):
        _ = NA(ChartPoint).price
    with pytest.raises(NaFieldError, match="field 'v'"):
        _ = Outer().inner.v


def __test_writing_a_field_of_an_na_object_raises__():
    """A field write on an na object is the same error, not a raw AttributeError"""
    na_inner = NA(Inner)
    with pytest.raises(NaFieldError, match="field 'v'"):
        na_inner.v = 2.0


def __test_copying_an_na_object_raises__():
    """Copying an na user-defined object is an error, an na drawing copies to na"""
    with pytest.raises(NaFieldError, match="cannot be copied"):
        udt_copy(NA(Inner))
    na_line = NA(Line)
    assert udt_copy(na_line) is na_line


def __test_copying_an_na_chart_point_raises_on_both_spellings__():
    """``p.copy()`` and ``chart.point.copy(p)`` on an na point are the same error"""
    with pytest.raises(NaFieldError, match="cannot be copied"):
        udt_copy(NA(ChartPoint))
    with pytest.raises(NaFieldError, match="cannot be copied"):
        chart.point.copy(NA(ChartPoint))


def __test_a_field_named_type_is_a_field_too__():
    """A field called ``type`` raises like any other and leaves the shared na intact"""
    na_kind = NA(Kind)
    with pytest.raises(NaFieldError, match="field 'type'"):
        _ = na_kind.type
    with pytest.raises(NaFieldError, match="field 'type'"):
        na_kind.type = 2.0
    assert NA(Kind) is na_kind
    with pytest.raises(NaFieldError, match="field 'v'"):
        _ = NA(Kind).v
    with pytest.raises(NaFieldError, match="cannot be copied"):
        udt_copy(NA(Kind))


def __test_other_attributes_of_an_na_stay_na__():
    """Names that are no field, and the fields of an na drawing, still answer na"""
    na_inner = NA(Inner)
    assert na_inner.not_a_field is na_inner
    na_line = NA(Line)
    assert na_line.x1 is na_line
    typeless = NA(None)
    assert typeless.v is typeless
    assert na_inner.type is Inner and na_line.type is Line and typeless.type is None


def __test_an_na_is_read_only__():
    """No attribute of an interned na can be rebound, so its declared type stays"""
    na_line = NA(Line)
    with pytest.raises(AttributeError):
        na_line.type = Inner
    assert na_line.type is Line
