"""
@pyne

Regression test for ``strategy.cancel_all()`` cancelling a MARKET entry.

A market ``strategy.entry()`` is queued for the next bar's open and lives in
the position's separate market-order dict, not only in the entry book. A
``cancel_all()`` later on the same bar must reach it too, otherwise the entry
still fills at the next open.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Cancel All Market Entry",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('M', strategy.long)
        strategy.cancel_all()


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
def __test_cancel_all_removes_pending_market_entry__(script_path, module_key):
    """A market entry cancelled on its placement bar must never fill.

    Buggy code path: ``_cancel_all_orders()`` clears the entry/exit books but
    leaves ``market_orders`` populated, so the entry fills at bar 1's open.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(4)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    for _candle, _plot, _new_closed in runner.run_iter():
        pass

    position = runner.script.position
    assert position.size == 0.0, (
        f"Expected flat position after cancel_all(), got size={position.size} -- "
        "a market entry survived strategy.cancel_all()."
    )
    assert position.closed_trades_count == 0
