"""
@pyne

COOF regression script: ``ta.valuewhen`` reads the bar index of the first two
occurrences of a condition that is true on bar 0 and on the fill bar 1.
"""
from pynecore.lib import bar_index, plot, script, strategy, ta
from pynecore.types import IBPersistent


@script.strategy(
    "COOF valuewhen ring",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_order_fills=True,
)
def main():
    # A varip slot is outside the COOF rollback, so it counts body executions,
    # not bars.
    execs: IBPersistent[int] = 0
    execs += 1

    cond = bar_index <= 1

    # The close can only be placed once the fill is visible, so it is placed in
    # the re-execution of the fill bar itself -- which is what makes bar 1 run
    # more than once.
    if bar_index == 0:
        strategy.entry('Long', strategy.long)
    if strategy.position_size > 0:
        strategy.close('Long')

    plot(ta.valuewhen(cond, bar_index, 0), 'occ0')
    plot(ta.valuewhen(cond, bar_index, 1), 'occ1')
    plot(ta.valuewhen(cond, bar_index, 2), 'occ2')
    plot(execs, 'total_execs')
