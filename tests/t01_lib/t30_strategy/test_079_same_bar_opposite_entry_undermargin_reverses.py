"""
@pyne

Guard test: a same-bar opposite pair that CAN be margined still reverses.

The same-bar first-leg keep is a MARGIN rule, not an order-priority rule. When
both legs of the flip fit within equity (small position relative to capital),
TradingView reverses on the same bar — the second entry flips the first. A
live TradingView probe on BINANCE:BTCUSDT confirmed a same-bar 0.8 BTC pair
(~47% of equity per leg) reverses, while 0.9 BTC (~55%) does not. This guards
against over-applying the same-bar keep as a blanket "first wins".
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Opposite Entry Under-Margin",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long)
        strategy.entry('B', strategy.short)
    if bar_index == 3:
        strategy.close_all(comment='flat')


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
def __test_same_bar_opposite_under_margin_reverses__(script_path, module_key):
    """
    ``A`` (long, qty 1 @ price 100 = $100, far below equity) fills, then ``B``
    (short) reverses it on the same bar — both legs fit within equity. ``A``
    closes via the flip and the position ends short until close_all.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200
    rows = [(100.0, 100.0, 100.0, 100.0) for _ in range(6)]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    closed_trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        closed_trades.extend(new_closed)

    # Reversal fits within margin: ``A`` (long) closes at bar-3 open when ``B``
    # flips short, then ``B`` (short) closes at close_all.
    assert len(closed_trades) == 2, [t.entry_id for t in closed_trades]
    assert closed_trades[0].entry_id == 'A'
    assert closed_trades[0].size > 0.0  # long leg
    assert closed_trades[1].entry_id == 'B'
    assert closed_trades[1].size < 0.0  # short leg (reversal opened it)
