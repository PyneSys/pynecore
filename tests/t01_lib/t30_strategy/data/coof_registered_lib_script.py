"""
@pyne

COOF regression script: a strategy importing a library whose entry runs on
every bar next to the strategy's ``main``. Both entry points must get their
own root vector in the runner — a root-key collision would detach main's root
and silently break the calc_on_order_fills var rollback.
"""
from pynecore.lib import plot, script, strategy
from pynecore.types import Persistent

import coof_registered_lib  # noqa: F401


@script.strategy(
    "COOF Registered Lib",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    calc_on_order_fills=True,
)
def main():
    var_exec: Persistent[int] = 0
    var_exec += 1

    if strategy.position_size == 0:
        strategy.entry('Long', strategy.long)

    plot(var_exec, 'var_exec')
