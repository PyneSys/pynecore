"""
@pyne

Regression test for ``strategy.cancel_all()`` clearing every pending order.

Two pending entries are placed and then cleared via ``strategy.cancel_all()``
before any can fill. No trades must result.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Cancel All",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=2,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long, stop=110.0)
        strategy.entry('B', strategy.long, stop=120.0)

    if bar_index == 1:
        strategy.cancel_all()


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_cancel_all_removes_every_pending_entry__(script_path, module_key):
    """``cancel_all()`` on bar 1 prevents both pending stop entries from ever filling.

    After ``cancel_all()`` on bar 1, neither stop entry must fill, even
    though bar 2's high=125 would trigger both stops (110 and 120). Buggy
    code path: pending orders survive, two trades open."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200

    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=100.0, high=100.5, low=99.5, close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=100.0, high=125.0, low=99.5, close=125.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=125.0, high=125.5, low=124.5, close=125.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)

    opened_entries = []
    for _candle, _plot, _new_closed in runner.run_iter():
        position = runner.script.position
        for order in list(position.open_trades):
            if order.entry_id not in opened_entries:
                opened_entries.append(order.entry_id)

    assert opened_entries == [], (
        f"Expected no entries to fill after cancel_all(), got {opened_entries} — "
        "strategy.cancel_all() regression."
    )
    assert position.size == 0.0, f"Expected flat position, got size={position.size}"
