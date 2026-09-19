"""
A discarded run's table edits must be rolled back like any other drawing change.

A table keeps its cells in a dict that the ``table.cell*`` functions change in
place: new cells are added, cleared ones removed, and the cell setters write the
fields of the same ``TableCell`` objects. A snapshot that only held the dict by
reference restored none of it.
"""


def __test_drawing_snapshot_rolls_back_table_cells__():
    """ Restoring a snapshot undoes added, cleared, edited and merged table cells """
    from pynecore.core.drawing_snapshot import DrawingSnapshot
    from pynecore.lib import position, table

    tb = table.new(position.top_right, 3, 3)
    try:
        table.cell(tb, 0, 0, "a")
        table.cell(tb, 1, 0, "b")
        kept = tb.cells[(0, 0)]

        snapshot = DrawingSnapshot()
        snapshot.save()

        table.cell_set_text(tb, 0, 0, "changed")
        table.cell(tb, 2, 2, "new")
        table.clear(tb, 1, 0)
        table.merge_cells(tb, 0, 0, 0, 1)

        snapshot.restore()

        assert {key: (cell.text, cell.is_merged) for key, cell in tb.cells.items()} == {
            (0, 0): ("a", False),
            (1, 0): ("b", False),
        }
        # The same cell object comes back, not a copy of it
        assert tb.cells[(0, 0)] is kept
    finally:
        table.delete(tb)
