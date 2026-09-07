"""
An ``na`` table dimension or coordinate is 0.

``lib/table.py`` truncates every dimension and coordinate at its consuming slot
(see ``test_011_table_int_typed_coords``), and that truncation went through a
bare ``int()``, which ``NA.__int__`` answers with a ``TypeError``. From PyneCore
v6.9.0 the Pine-docs construct ``table.new(position.top_left, na, na)`` -- and
every cell call reached with an na coordinate -- crashed the script (issue #79).

MEASURED on TradingView (BITSTAMP:BTCUSD@60):

- ``table.new(pos, na, na)`` CONSTRUCTS; only a later ``table.cell`` fails, with
  RE10039 "Column 0 is out of table bounds, number of columns is 0".
- ``table.new(pos, 5, na)`` fails on the ROW side instead (RE10040, rows 0), so
  the coercion is per-argument, not a poisoned table.
- ``table.cell(t, na, na, ...)`` on a 2x2 table SUCCEEDS -- it addresses cell
  (0, 0) -- and on a 0x0 table raises the same RE10039, so an na coordinate is
  bounds-checked as 0 rather than skipped.
- ``table.merge_cells``/``table.clear`` with all-na bounds run without error.

PyneCore does not bounds-check table coordinates at all (``Table.cells`` is an
unbounded dict), so the out-of-bounds abort is out of scope here; what these
tests pin is the na -> 0 coercion.

Both na representations are exercised: bare ``na`` in a script is the typeless
``NA`` object (``NA(None)``, ``lib._na_none``), whose ``__int__`` raises
``TypeError``, while an int-typed na travels as a native ``nan``
(``NA(int)`` is interned to one), whose ``int()`` raises ``ValueError``. Only the first is what
issue #79 hit, and only ``NA`` answers ``False`` to ``!=`` as well as ``==``.
"""
import pytest

from pynecore.lib import table, position
from pynecore.types.na import NA

# Bare ``na`` in a script, and an int-typed na, which is a native nan
NAS = (NA(None), NA(int))


@pytest.mark.parametrize('na', NAS)
def __test_new_na_dimensions_build_an_empty_table__(na):
    """na columns/rows are 0, and the constructor does not raise"""
    t = table.new(position.top_left, na, na)
    assert t.columns == 0 and t.rows == 0


@pytest.mark.parametrize('na', NAS)
def __test_new_coerces_each_dimension_on_its_own__(na):
    """An na row does not drag the column count with it"""
    t = table.new(position.top_left, 5, na)
    assert t.columns == 5 and t.rows == 0


@pytest.mark.parametrize('na', NAS)
def __test_cell_na_coordinate_addresses_cell_zero__(na):
    """An na column/row writes cell (0, 0)"""
    t = table.new(position.top_left, 2, 2)
    table.cell(t, na, na, "na")
    assert t.get_cell(0, 0).text == "na"
    assert list(t.cells) == [(0, 0)]


@pytest.mark.parametrize('na', NAS)
def __test_cell_setters_share_that_cell__(na):
    """The setters resolve the same (0, 0) an na coordinate names"""
    t = table.new(position.top_left, 2, 2)
    table.cell(t, 0, 0, "text")
    table.cell_set_tooltip(t, na, na, "tip")
    table.cell_set_text_color(t, na, na, None)
    assert t.get_cell(0, 0).tooltip == "tip"
    assert list(t.cells) == [(0, 0)]


@pytest.mark.parametrize('na', NAS)
def __test_clear_and_merge_accept_na_bounds__(na):
    """The range walks consume na bounds as 0 instead of raising"""
    t = table.new(position.top_left, 2, 2)
    table.cell(t, 0, 0, "00")
    table.cell(t, 1, 1, "11")
    table.merge_cells(t, na, na, na, na)
    assert t.get_cell(0, 0).is_merged
    assert t.get_cell(0, 0).merge_end_col == 0

    table.clear(t, na, na)
    assert (0, 0) not in t.cells and (1, 1) in t.cells
