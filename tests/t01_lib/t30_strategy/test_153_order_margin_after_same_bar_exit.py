"""
@pyne

A ``strategy.order`` that fits against the bar-start position fills even when a
same-bar exit has flattened that position first.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, 100 % margin, a 0.99x-equity long with
an exit stop above a ``strategy.order`` sell stop, both crossed on one bar): the sell
fills at 1.02x and 1.8x of equity (246/246 each) and the margin call trims it right
away -- at the low the descending leg reached, not at the close. At 2.5x, where even
the net position against the bar-start long cannot be margined, it never fills
(275/275).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Order Margin After Same Bar Exit",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0 or bar_index == 10:
        strategy.entry('L', strategy.long, qty=9)
    if bar_index == 1 or bar_index == 11:
        strategy.exit('X', 'L', stop=99.7)
    if bar_index == 1:
        strategy.order('S', strategy.short, qty=16.2, stop=99.5)
    if bar_index == 11:
        strategy.order('S', strategy.short, qty=25, stop=99.5)
    if bar_index == 5 or bar_index == 15:
        strategy.cancel('S')
        strategy.close_all()


def __test_helper_make_syminfo():
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


def __test_helper_run(script_path, module_key, rows):
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
    return trades


def __test_helper_shape(trades):
    return [(t.entry_id, t.exit_id, t.exit_comment or None, round(abs(t.size), 6), round(t.entry_price, 2),
             round(t.exit_price, 2)) for t in trades]


# noinspection PyShadowingNames
def __test_order_fits_against_bar_start_position__(script_path, module_key):
    """
    The 16.2 sell (9 long at bar start -> 7.2 net) fills after the exit and the margin
    call takes all of it at the low; the 25 sell (16 net) is rejected.
    """
    flat = (100.0, 100.05, 99.95, 100.0)
    drop = (100.0, 100.05, 99.0, 99.6)
    rows = [flat, flat, drop] + [flat] * 9 + [drop] + [flat] * 5
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('L', 'X', None, 9.0, 100.0, 99.7),
        ('S', None, 'Margin call', 16.2, 99.5, 99.0),
        ('L', 'X', None, 9.0, 100.0, 99.7),
    ], shape
