"""
@pyne

Regression test for ``strategy.cancel(id)`` matching exit ids, not just entry ids.

Pine ``strategy.cancel('TP1')`` must cancel a partial-TP placed as
``strategy.exit('TP1', from_entry='Long', ...)``. Before the fix,
``Position._remove_order_by_id`` only matched ``exit_order.order_id``
(= ``from_entry``), so cancelling by exit id was a no-op and TP1 still fired.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Cancel By Exit Id",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('Long', strategy.long)

    if bar_index == 1:
        strategy.exit('TP1', from_entry='Long', qty=1, limit=110.0)
        strategy.exit('TP2', from_entry='Long', qty=1, limit=120.0)

    # Cancel TP1 by its exit id before TP1's price is reached.
    if bar_index == 2:
        strategy.cancel('TP1')


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_cancel_by_exit_id_removes_only_tp1__(script_path, module_key):
    """
    After cancelling TP1 by its exit id on bar 2, only TP2 (limit 120) must fire.

    Buggy code path: cancel was no-op, TP1 (limit 110) still fires on bar 3 high=115,
    producing 2 trades instead of 1.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200

    # Bar 0: entry signal (fills at bar 1 open)
    # Bar 1: entry fills at 100, both exits placed
    # Bar 2: cancel('TP1') runs; price stays low so neither exit can fill
    # Bar 3: high=115 — TP1 would fire here if not cancelled (limit 110)
    # Bar 4: high=125 — TP2 fires (limit 120)
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=100.0, high=115.0, low=100.0, close=115.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60, open=115.0, high=125.0, low=115.0, close=125.0, volume=100.0),
        OHLCV(timestamp=base_ts + 5 * 60, open=125.0, high=125.5, low=124.5, close=125.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, (
        f"Expected 1 closed trade (only TP2 — TP1 was cancelled), got {len(trades)} — "
        "strategy.cancel(exit_id) regression."
    )
    assert trades[0].exit_id == 'TP2', f"Expected TP2, got {trades[0].exit_id}"
    assert trades[0].exit_price == 120.0, f"TP2 price: {trades[0].exit_price}"
