"""
@pyne

COOF regression script: the body draws one line per execution. A fill bar's
body runs several times, but only the last run counts, so the bar must still
leave exactly one line behind.
"""
from pynecore.lib import array, line, plot, script, strategy, bar_index

# A module-level global is outside the slot scheme, so the COOF rollback does
# not touch it -- it counts body executions, not bars.
_execs: list[int] = []


@script.strategy(
    "COOF Drawing Rollback",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_order_fills=True,
    max_lines_count=500,
)
def main():
    _execs.append(bar_index)
    line.new(bar_index, 0.0, bar_index, 1.0)

    # The close can only be placed once the fill is visible, so it is placed in
    # the re-execution of the fill bar itself -- which is what makes the bar run
    # more than once.
    if bar_index == 0:
        strategy.entry('Long', strategy.long)
    if strategy.position_size > 0:
        strategy.close('Long')

    plot(array.size(line.all), 'lines')
    plot(len(_execs), 'total_execs')
