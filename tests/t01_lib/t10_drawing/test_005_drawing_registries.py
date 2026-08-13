"""
@pyne
"""
from pynecore.lib import (
    script, plot, array, chart, line, box, label, linefill, polyline, table,
    bar_index, close, high, low, color, position
)


@script.indicator(title="Drawing Registries", shorttitle="registries")
def main():
    # Every ``<drawing>.all`` registry is an array, so the array namespace reads it
    before = (array.size(box.all) + array.size(label.all) + array.size(line.all)
              + array.size(linefill.all) + array.size(polyline.all) + array.size(table.all))

    lb = label.new(bar_index, close, "l")
    bx = box.new(bar_index, high, bar_index, low)
    l1 = line.new(bar_index, high, bar_index, low)
    l2 = line.new(bar_index, close, bar_index, close)
    lf = linefill.new(l1, l2, color.red)
    pl = polyline.new([chart.point.from_index(bar_index, high),
                       chart.point.from_index(bar_index, low)])
    tb = table.new(position.top_right, 1, 1)

    plot(before, "before")
    plot(array.size(label.all), "labels")
    plot(array.size(box.all), "boxes")
    plot(array.size(line.all), "lines")
    plot(array.size(linefill.all), "linefills")
    plot(array.size(polyline.all), "polylines")
    plot(array.size(table.all), "tables")

    # The registry is a real array: its last element is the object just created
    plot(1 if array.get(label.all, array.size(label.all) - 1) == lb else 0, "last_label")
    plot(1 if array.get(box.all, array.size(box.all) - 1) == bx else 0, "last_box")
    plot(1 if array.get(linefill.all, array.size(linefill.all) - 1) == lf else 0, "last_linefill")
    plot(1 if array.get(polyline.all, array.size(polyline.all) - 1) == pl else 0, "last_polyline")
    plot(1 if array.get(table.all, array.size(table.all) - 1) == tb else 0, "last_table")

    # Every registry read is a snapshot: emptying the returned array must leave
    # the runtime's own registry alone, only ``<drawing>.delete`` may shrink it
    array.clear(box.all)
    array.clear(label.all)
    array.clear(line.all)
    array.clear(linefill.all)
    array.clear(polyline.all)
    array.clear(table.all)
    plot(array.size(box.all) + array.size(label.all) + array.size(line.all)
         + array.size(linefill.all) + array.size(polyline.all) + array.size(table.all),
         "after_clear")

    label.delete(lb)
    box.delete(bx)
    line.delete(l1)
    line.delete(l2)
    linefill.delete(lf)
    polyline.delete(pl)
    table.delete(tb)

    # Deleting every object empties every registry again
    plot(array.size(box.all) + array.size(label.all) + array.size(line.all)
         + array.size(linefill.all) + array.size(polyline.all) + array.size(table.all),
         "after")


def __test_drawing_registries__(csv_reader, runner, log):
    """ Every ``<drawing>.all`` registry reads as an array and tracks new/delete """
    bars = 0
    with csv_reader('chart_point.csv', subdir="data") as cr:
        for i, (_candle, _plot) in enumerate(runner(cr).run_iter()):
            bars += 1
            assert _plot['before'] == 0, f"bar {i}: registries not empty at bar start"
            assert _plot['after'] == 0, f"bar {i}: registries not empty after delete"
            assert _plot['after_clear'] == 7, \
                f"bar {i}: clearing a registry array changed the registry ({_plot['after_clear']})"
            assert _plot['labels'] == 1, f"bar {i}: label.all size {_plot['labels']}"
            assert _plot['boxes'] == 1, f"bar {i}: box.all size {_plot['boxes']}"
            assert _plot['lines'] == 2, f"bar {i}: line.all size {_plot['lines']}"
            assert _plot['linefills'] == 1, f"bar {i}: linefill.all size {_plot['linefills']}"
            assert _plot['polylines'] == 1, f"bar {i}: polyline.all size {_plot['polylines']}"
            assert _plot['tables'] == 1, f"bar {i}: table.all size {_plot['tables']}"
            for name in ('last_label', 'last_box', 'last_linefill',
                         'last_polyline', 'last_table'):
                assert _plot[name] == 1, f"bar {i}: {name} is not the object just created"

    assert bars > 0, "no bars were run"
    log.info("Drawing registries verified on %d bars", bars)
