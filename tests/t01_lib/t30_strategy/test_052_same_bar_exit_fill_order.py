"""
@pyne

Regression test for the same-bar exit fill ORDER on a downward price walk.

Two stop legs share one entry: X_A (tight, stop 95) and X_B (wide, stop 90).
A single bar opens above both stops and trades down through both, so both fill
on that bar. The level nearest the open (X_A, 95) is reached FIRST in time, so it
must close first; TradingView lists same-bar exits in this intrabar order.

Before the fix the open->low walk iterated ``iter_orders(min_price=l, max_price=o)``
ASCENDING, filling the level FARTHEST from the open (X_B, 90) first -- the reverse
of the real price path. The walk now goes descending for the open->low leg
(nearest-the-open first), so the closed-trade order matches TradingView.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Exit Fill Order",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if bar_index == 1:
        strategy.exit('X_A', from_entry='L', qty=1, stop=95.0)
        strategy.exit('X_B', from_entry='L', qty=1, stop=90.0)


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_same_bar_stops_fill_nearest_open_first__(script_path, module_key):
    """
    Two stops hit on one down bar close in nearest-the-open order: X_A (95) then X_B (90).

    * bar 0: entry signal -> fills bar 1 open at 100.
    * bar 1: both stops placed (X_A 95, X_B 90).
    * bar 2: opens at 98 (above both stops, no gap), drops to a low of 88 (below
      both). Walking down from the open, the stop nearest the open (X_A, 95) is
      reached first, then X_B (90). Both fill on bar 2.

    Before the fix the down walk filled the farthest stop (X_B, 90) first, so the
    closed-trade order was reversed.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    rows = [
        # open,  high,  low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - entry fill @100, both stops placed
        (98.0,  98.5,  88.0, 92.0),   # bar 2 - down walk through both stops
        (92.0,  92.5,  91.5, 92.0),   # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, f"Expected 2 closed trades, got {len(trades)}"
    # Both fill on bar 2.
    for t in trades:
        assert t.exit_bar_index == 2, f"{t.exit_id} exit bar {t.exit_bar_index}"
    # Order: the stop nearest the open (X_A, 95) closes first.
    assert trades[0].exit_id == 'X_A', (
        f"expected X_A (stop 95, nearest the open) to close first, "
        f"got {trades[0].exit_id} first"
    )
    assert trades[1].exit_id == 'X_B', f"expected X_B second, got {trades[1].exit_id}"
    assert abs(trades[0].exit_price - 95.0) < 1e-9, f"X_A exit {trades[0].exit_price}"
    assert abs(trades[1].exit_price - 90.0) < 1e-9, f"X_B exit {trades[1].exit_price}"
