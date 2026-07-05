"""
@pyne

Regression test for a same-direction second entry that over-margins only because
a prior same-bar entry already filled (a pyramid stack).
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same Bar Stack Overmargin Fills",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
    pyramiding=10,
    margin_long=100,
    margin_short=0,
)
def main():
    if bar_index == 0:
        # Two same-direction market entries on the same bar. Each is sized at 100%
        # of equity, so each alone fits, but together they double the position.
        strategy.entry('A', strategy.long)
        strategy.entry('B', strategy.long)


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
def __test_same_bar_stack_fills_then_margin_call_trims__(script_path, module_key):
    """
    The second same-direction entry only over-margins because the first same-bar
    entry already filled. It cleared its placement-time margin check, so TV does
    NOT hard-reject it (unlike a fresh entry, see test_064): it fills and the
    bar-open margin call trims the doubled position. Both entries are visible in
    the ledger, matching TradingView's broker emulator.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    rows = [
        (100.00, 100.00, 100.00, 100.00),
        (100.00, 100.00, 100.00, 100.00),
        (100.00, 100.00, 100.00, 100.00),
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    closed_trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    # Both same-bar entries fill; the second is NOT pre-fill rejected. The bar-open
    # margin call then liquidates the whole doubled position (each entry sized at
    # 100% equity, so the 200% aggregate is fully unaffordable at 100% margin).
    assert [(t.entry_id, t.exit_comment) for t in closed_trades] == [
        ('A', 'Margin call'),
        ('B', 'Margin call'),
    ]
