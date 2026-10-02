"""
@pyne

A short opened on the descending leg is judged at the low after the leg, and at its
fill price too.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, mc8 probe, a sell stop filling right
after a same-bar exit of a long): when the account is short of margin at the low, the
cut is sized at the low; when the low is in surplus but the fill price was not, the
cut is sized at the fill price -- 4x the lots of the shortfall there -- and executed
at the low (16/16).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Margin Call At Fill Price",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=9)
    if bar_index == 1:
        strategy.exit('X', 'L', stop=99.7)
        strategy.order('S', strategy.short, qty=10.2, stop=99.5)
    if bar_index == 4:
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
def __test_cut_sized_at_fill_price_executed_at_low__(script_path, module_key):
    """
    Equity after the exit is 997.3; at the 99.5 fill the 10.2 short needs 1014.9, a
    17.6 shortfall = 1768 lots, cut 4x = 0.7072 at the low 98.5 (where the account is
    2.8 in surplus).
    """
    flat = (100.0, 100.05, 99.95, 100.0)
    rows = [flat, flat, (100.0, 100.05, 98.5, 99.6), flat, flat, flat, flat]
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('L', 'X', None, 9.0, 100.0, 99.7),
        ('S', None, 'Margin call', 0.7072, 99.5, 98.5),
        ('S', 'Close position order', None, 9.4928, 99.5, 100.0),
    ], shape
