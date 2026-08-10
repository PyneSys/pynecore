"""
@pyne

Regression test for exit slices sized against a pyramid add that never happens.

A same-direction market entry issued while the pyramiding limit is already
reached is dropped when the order is processed, without ever touching the
position. PyneCore still counted it as bound to the entry id, so a
``qty_percent`` leg re-issued on that bar reserved a share of the doubled size
and went on to close the WHOLE position. The inflated reservation is sticky
too: on the next bar the sibling leg finds nothing unreserved left and keeps its
own oversized share, so the bracket never recovers.

Measured on the CAPITALCOM:EURUSD 30m TradingView reference of the "TradingView
Alerts to MT4 MT5" strategy, whose entry condition fires again while the
position is already open: TV keeps the partial leg at half of the position that
actually exists.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Rejected Pyramid Add Exit Slice",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=100,
)
def main():
    if bar_index in (0, 2):
        strategy.entry('L', strategy.long)
    strategy.exit('XPart', 'L', qty_percent=50, limit=101.00)
    strategy.exit('X', 'L', stop=98.00)


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_rejected_pyramid_add_keeps_half_slice__(script_path, module_key):
    """
    The ``qty_percent=50`` leg must close half the position, not all of it.

    * bar 0: entry signal -> fills bar 1 open at 100.00 (qty 100).
    * bar 2: the same signal again -> a pyramid add the default ``pyramiding=0``
      rejects at bar 3's open, so the position stays at 100.
    * bar 4: the high reaches the 101.00 limit -> the partial leg closes 50 and
      the other 50 stays open.

    The pre-fix behavior sized the partial leg off 200 (open position plus the
    doomed add) and closed the full 100.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - entry signal
        (100.00, 100.50, 99.90, 100.40),   # bar 1 - entry fills at 100.00
        (100.40, 100.60, 100.20, 100.50),  # bar 2 - rejected pyramid signal
        (100.50, 100.70, 100.30, 100.60),  # bar 3 - the add is dropped at open
        (100.60, 101.20, 100.40, 101.00),  # bar 4 - the 101.00 limit fills
        (101.00, 101.10, 100.80, 100.90),  # bar 5 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, f"expected one partial exit, got {len(trades)}: {trades}"
    closed = trades[0]
    assert abs(closed.exit_price - 101.00) < 1e-9, f"exit_price={closed.exit_price}"
    assert abs(abs(closed.size) - 50.0) < 1e-9, (
        f"the partial leg must close half the position, got {closed.size}"
    )
    assert abs(runner.script.position.size - 50.0) < 1e-9, (
        f"the other half must stay open, position size is {runner.script.position.size}"
    )
