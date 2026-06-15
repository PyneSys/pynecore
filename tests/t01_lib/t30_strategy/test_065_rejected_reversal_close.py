"""
@pyne

Regression test for same-bar closes after a rejected reversal entry.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Rejected Reversal Close",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
    margin_long=100,
    margin_short=0,
)
def main():
    if bar_index == 0:
        strategy.entry('S', strategy.short)
    if bar_index == 2 and strategy.position_size < 0:
        strategy.entry('L', strategy.long)
        strategy.close('S', comment='SL Hit')
    if bar_index == 3 and strategy.position_size < 0:
        strategy.close('S', comment='SL Hit')


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
def __test_rejected_reversal_does_not_fill_same_bar_close__(script_path, module_key):
    """
    A same-bar ``strategy.close`` does not become a fallback fill when the
    redundant reversal entry is rejected at the next open.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    rows = [
        (100.00, 100.00, 100.00, 100.00),
        (100.00, 100.00, 99.50, 100.00),
        (100.00, 100.00, 99.50, 100.00),
        (100.01, 100.01, 99.50, 100.00),
        (100.00, 100.00, 99.50, 100.00),
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    closed_trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    assert len(closed_trades) == 1
    trade = closed_trades[0]
    assert trade.exit_comment == 'SL Hit'
    assert trade.exit_bar_index == 4
    assert trade.exit_price == 100.00
