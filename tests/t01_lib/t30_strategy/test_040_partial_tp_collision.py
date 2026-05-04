"""
@pyne

Regression test for the exit_orders dict-key collision bug.

Two ``strategy.exit()`` calls with the same ``from_entry`` but different ``id``
(classic partial take-profit pattern: TP1 at +10, TP2 at +20, each closing half
the position) must result in two independent exit orders that both fire.

Before the fix, ``Position._add_order`` keyed ``exit_orders`` by
``order.order_id`` (= ``from_entry`` = "Long"), so the second call silently
evicted the first from both ``exit_orders`` and the orderbook — only one TP
ever fired and the remaining half stayed open forever.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Partial TP Collision",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=0,
)
def main():
    # Single entry on bar 0 (fills at bar 1 open).
    if bar_index == 0:
        strategy.entry('Long', strategy.long)

    # Place both partial-TPs once, on the bar after the fill, so each is a
    # standalone exit order — not a per-bar replacement that would resurrect
    # TP1 every tick after it had already fired.
    if bar_index == 1:
        strategy.exit('TP1', from_entry='Long', qty=1, limit=110.0)
        strategy.exit('TP2', from_entry='Long', qty=1, limit=120.0)


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider.get_opening_hours_and_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_two_exits_same_from_entry_both_fire__(script_path, module_key):
    """
    Two strategy.exit() calls with same from_entry, different ids and qty each = 1
    must produce two distinct closed trades — one at limit 110, one at limit 120.

    On the buggy code path: TP2 evicts TP1 from exit_orders, only TP2 ever fires,
    only 1 unit of the 2-unit position closes, total closed trades = 1.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    # Bar 0: flat — emit entry signal (market order queued for next open)
    # Bar 1: entry fills at open=100; both exits placed at end of bar
    # Bar 2: high=115 → TP1 (limit 110) fills, position size goes 2 -> 1
    # Bar 3: high=125 → TP2 (limit 120) fills, position closes
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60,  open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60,  open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60,  open=100.0, high=115.0, low=100.0, close=115.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60,  open=115.0, high=125.0, low=115.0, close=125.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60,  open=125.0, high=125.5, low=124.5, close=125.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, (
        f"Expected 2 closed trades (TP1 + TP2), got {len(trades)} — "
        "exit_orders dict-key collision likely regressed."
    )

    exit_ids = sorted(t.exit_id for t in trades)
    assert exit_ids == ['TP1', 'TP2'], f"Expected exit ids [TP1, TP2], got {exit_ids}"

    by_id = {t.exit_id: t for t in trades}
    assert by_id['TP1'].exit_price == 110.0, f"TP1 exit price: {by_id['TP1'].exit_price}"
    assert by_id['TP2'].exit_price == 120.0, f"TP2 exit price: {by_id['TP2'].exit_price}"
    assert by_id['TP1'].size == 1.0, f"TP1 size: {by_id['TP1'].size}"
    assert by_id['TP2'].size == 1.0, f"TP2 size: {by_id['TP2'].size}"
