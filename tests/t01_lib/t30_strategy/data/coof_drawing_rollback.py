"""
@pyne

COOF regression script: the body draws one line per execution. A fill bar's
body runs several times, but only the last run counts, so the bar must still
leave exactly one line behind.
"""
from pynecore.lib import array, line, plot, script, strategy, bar_index
from pynecore.types import IBPersistent


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
    # A varip slot is outside the COOF rollback, so it counts body executions,
    # not bars.
    execs: IBPersistent[int] = 0
    execs += 1
    line.new(bar_index, 0.0, bar_index, 1.0)

    # The close can only be placed once the fill is visible, so it is placed in
    # the re-execution of the fill bar itself -- which is what makes the bar run
    # more than once.
    if bar_index == 0:
        strategy.entry('Long', strategy.long)
    if strategy.position_size > 0:
        strategy.close('Long')

    plot(array.size(line.all), 'lines')
    plot(execs, 'total_execs')
