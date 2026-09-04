"""
``table`` coordinates are int-TYPED, so the façade truncates them.

Pine's ``int`` is a static type only: ``(R + z) / 8`` is int-typed and carries
the value 1.75. ``lib/matrix.py`` normalizes every coordinate with ``int()``,
``lib/table.py`` did not -- a fractional column became a distinct dictionary key
in ``Table.get_cell`` and made ``Table.clear_cells``' ``range()`` raise.

MEASURED on TradingView (FX:EURUSD@60, ``d = (R + z) / 8`` = 1.75):
``table.clear(t, d, d, d + 1, d + 1)`` runs; PyneCore raised
``TypeError: 'float' object cannot be interpreted as an integer``.
"""
from pynecore.lib import table, position


def _new_table():
    return table.new(position.top_right, 4, 4)


def __test_cell_lands_on_the_truncated_coordinate__():
    """A fractional column/row addresses the same cell as its truncated pair"""
    t = _new_table()
    table.cell(t, 1.75, 2.75, "frac")
    assert t.get_cell(1, 2).text == "frac"
    # ... and does not open a second cell beside it
    assert list(t.cells) == [(1, 2)]


def __test_cell_setters_share_that_cell__():
    """The setters resolve the same truncated coordinate as ``table.cell``"""
    t = _new_table()
    table.cell(t, 14 / 8, 14 / 8, "text")
    table.cell_set_tooltip(t, 14 / 8, 14 / 8, "tip")
    assert t.get_cell(1, 1).tooltip == "tip"
    assert list(t.cells) == [(1, 1)]


def __test_clear_and_merge_accept_fractional_ranges__():
    """The range walks consume their bounds as integers instead of raising"""
    t = _new_table()
    for column in range(3):
        for row in range(3):
            table.cell(t, column, row, f"{column}{row}")
    table.merge_cells(t, 0.5, 0.5, 1.75, 1.75)
    assert t.get_cell(0, 0).is_merged
    assert t.get_cell(0, 0).merge_end_col == 1

    table.clear(t, 1.75, 1.75, 2.75, 2.75)
    assert (1, 1) not in t.cells and (2, 2) not in t.cells
    assert (0, 0) in t.cells


def __test_new_truncates_the_dimensions__():
    """The column and row counts are counts, so they truncate"""
    t = table.new(position.top_left, 7 / 2, 9 / 2)
    assert t.columns == 3 and t.rows == 4
