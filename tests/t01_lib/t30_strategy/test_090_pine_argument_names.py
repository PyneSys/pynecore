"""
Regression: the parameter NAMES of the strategy.* order commands must be Pine's.

A compiled Pine script that uses a named argument emits Pine's own keyword, so a
parameter spelled differently from TradingView turns an ordinary call into a
``TypeError`` that halts the script. The names below were verified against the
TradingView compiler.

The commands run with ``lib._lib_semaphore`` set, so each one returns before it
touches any strategy state -- keyword binding is checked by the call itself,
which is exactly where a wrong parameter name raises.
"""
from contextlib import contextmanager

import pytest

from pynecore import lib
from pynecore.lib import strategy


@contextmanager
def _suppressed():
    """Make every strategy command an immediate no-op, keeping argument binding live."""
    # noinspection PyProtectedMember
    lib._lib_semaphore = True
    try:
        yield
    finally:
        # noinspection PyProtectedMember
        lib._lib_semaphore = False


def __test_strategy_order_commands_accept_disable_alert__():
    """entry/order/close/close_all name their last parameter ``disable_alert``"""
    with _suppressed():
        strategy.entry(id="E", direction=strategy.long, disable_alert=True)
        strategy.entry(id="E", direction=strategy.long, qty=1, limit=1.0, stop=1.0,
                       oca_name="o", oca_type=strategy.oca.none, comment="c",
                       alert_message="a", disable_alert=True)
        strategy.order(id="E", direction=strategy.long, disable_alert=True)
        strategy.close(id="E", disable_alert=True)
        strategy.close(id="E", comment="c", qty=1, qty_percent=100, alert_message="a",
                       immediately=False, disable_alert=True)
        strategy.close_all(disable_alert=True)
        strategy.close_all(comment="c", alert_message="a", immediately=False,
                           disable_alert=True)


def __test_strategy_exit_takes_pine_parameters_without_oca_type__():
    """strategy.exit takes Pine's 21 parameters in Pine's order and has no ``oca_type``"""
    with _suppressed():
        strategy.exit(id="X", from_entry="E", qty=1, qty_percent=100, profit=10,
                      limit=1.0, loss=10, stop=1.0, trail_price=1.0, trail_points=10,
                      trail_offset=10, oca_name="o", comment="c", comment_profit="cp",
                      comment_loss="cl", comment_trailing="ct", alert_message="a",
                      alert_profit="ap", alert_loss="al", alert_trailing="at",
                      disable_alert=True)
        # The same arguments fully positional -- guards the parameter ORDER
        strategy.exit("X", "E", 1, 100, 10, 1.0, 10, 1.0, 1.0, 10, 10, "o", "c", "cp",
                      "cl", "ct", "a", "ap", "al", "at", True)
        # Pine's strategy.exit has no oca_type; its legs always form a reduce group
        with pytest.raises(TypeError):
            # noinspection PyArgumentList
            strategy.exit(id="X", from_entry="E", profit=10,  # type: ignore[call-arg]
                          oca_type=strategy.oca.none)
