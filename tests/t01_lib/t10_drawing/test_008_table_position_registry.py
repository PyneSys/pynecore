"""
TradingView keeps at most one table per position.

Measured on CAPITALCOM:EURUSD with ``plot(array.size(table.all))``: a non-``var``
``table.new`` at one position keeps the count at 1 over 23130 bars, two positions
keep it at 2, and ``table.set_position`` onto an occupied position drops it from
2 to 1 -- the moved table survives and the previous occupant is gone.
"""


def __test_helper_clear_registry():
    from pynecore.lib import table
    table._registry.clear()


def __test_table_new_replaces_the_table_at_the_same_position__():
    """ table.new at an occupied position evicts the previous table """
    from pynecore.lib import position, table

    try:
        first = table.new(position.top_right, 1, 1)
        second = table.new(position.top_right, 1, 1)
        other = table.new(position.bottom_left, 1, 1)

        assert table._registry == [second, other]
        assert all(tbl is not first for tbl in table._registry)

        # An evicted handle stays usable: no raise, no resurrection
        table.cell(first, 0, 0, "orphan")
        table.delete(first)
        assert table._registry == [second, other]
    finally:
        __test_helper_clear_registry()


def __test_table_new_registry_stays_bounded__():
    """ Repeated table.new at one position does not grow the registry """
    from pynecore.lib import position, table

    try:
        for _ in range(100):
            tbl = table.new(position.middle_center, 2, 2)
            table.cell(tbl, 0, 0, "x")
        assert len(table._registry) == 1
    finally:
        __test_helper_clear_registry()


def __test_set_position_evicts_the_previous_occupant__():
    """ Moving a table onto an occupied position deletes the table already there """
    from pynecore.lib import position, table

    try:
        occupant = table.new(position.top_left, 1, 1)
        mover = table.new(position.top_right, 1, 1)

        table.set_position(mover, position.top_left)

        assert table._registry == [mover]
        assert mover.position == position.top_left

        # Moving a table onto its own position keeps it registered
        table.set_position(mover, position.top_left)
        assert table._registry == [mover]

        table.delete(occupant)
        assert table._registry == [mover]
    finally:
        __test_helper_clear_registry()


def __test_snapshot_restores_the_registry_after_a_replacement__():
    """ A discarded run's table.new replacement is rolled back """
    from pynecore.core.drawing_snapshot import DrawingSnapshot
    from pynecore.lib import position, table

    try:
        original = table.new(position.bottom_right, 1, 1)
        table.cell(original, 0, 0, "kept")

        snapshot = DrawingSnapshot()
        snapshot.save()

        table.new(position.bottom_right, 1, 1)
        assert table._registry != [original]

        snapshot.restore()

        assert table._registry == [original]
        assert original.cells[(0, 0)].text == "kept"
    finally:
        __test_helper_clear_registry()
