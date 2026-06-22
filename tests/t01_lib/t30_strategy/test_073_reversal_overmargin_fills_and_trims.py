"""
@pyne

Regression test for the new leg of a reversal that overshoots margin at fill.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Reversal Overmargin Fills And Trims",
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
        # Close the short FIRST, then open the opposite (long) leg on the same bar.
        strategy.close('S', comment='Reverse')
        strategy.entry('L', strategy.long)


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
def __test_reversal_new_leg_fills_then_margin_call_trims__(script_path, module_key):
    """
    When a same-bar ``strategy.close`` flattens the old position before the
    opposite ``strategy.entry`` is processed, the new leg is NOT pre-fill
    rejected like a fresh entry. TradingView fills it and the bar-open margin
    call trims the over-margin excess to a viable remainder.
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

    position = runner.script.position

    # The new long leg FILLS (not rejected) and survives with a trimmed remainder:
    # full 1000-contract long minus exactly one whole contract liquidated by the
    # bar-open margin call. A fresh entry in the same margin state would be rejected
    # (see test_064); the same-bar close before it makes this a reversal leg.
    assert position.size == 999.0

    # Ledger: the short is reversed (closed), and the over-margin excess of the new
    # long is removed by a single whole-contract margin call at the fill price.
    assert [(t.entry_id, t.exit_comment, t.size, t.exit_price) for t in closed_trades] == [
        ('S', 'Reverse', -1000.0, 100.01),
        ('L', 'Margin call', 1.0, 100.01),
    ]
