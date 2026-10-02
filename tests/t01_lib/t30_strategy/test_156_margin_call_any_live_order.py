"""
@pyne

Under ``close_entries_rule='ANY'`` a margin call cuts every open trade, then stays a
live order that cuts each later trade once, until the position goes flat; a
``strategy.close(id, qty_percent)`` takes the percent of the entry's original quantity.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, mc4-mc7 probes): the margin-call size
comes off EACH open trade (579/579), every trade opened later loses the same size at
its first margin checkpoint -- the bar open for a market entry -- and a close_all ends
it; ``qty_percent=10`` closes 10 % of the ORIGINAL quantity (217/217), while FIFO
closes 10 % of what is still open.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Margin Call ANY Live Order",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=5,
    margin_long=50,
    close_entries_rule="ANY",
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long, qty=15)
    if bar_index == 1:
        strategy.entry('B', strategy.long, qty=4.9)
    if bar_index == 4:
        strategy.entry('C', strategy.long, qty=1)
    if bar_index == 5:
        strategy.close('A', qty_percent=10)
    if bar_index == 6:
        strategy.close_all()
    if bar_index == 7:
        strategy.entry('D', strategy.long, qty=1)
    if bar_index == 9:
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
def __test_live_margin_call_and_original_percent__(script_path, module_key):
    """
    At the 98.9 low the account is 5.945 short = 1202 lots, cut 4x = 0.4808 from A and
    from B. C, entered later, loses 0.4808 at its 101 open fill; A's 10 % close takes
    1.5 (of 15, not of 14.5192); D, after the close_all, keeps its whole quantity.
    """
    flat = (100.0, 100.05, 99.95, 100.0)
    high = (101.0, 101.05, 100.95, 101.0)
    rows = [flat, flat, flat, (100.0, 100.05, 98.9, 99.5), (99.5, 101.0, 99.4, 101.0),
            high, high, high, high, high, high]
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('A', None, 'Margin call', 0.4808, 100.0, 98.9),
        ('B', None, 'Margin call', 0.4808, 100.0, 98.9),
        ('C', None, 'Margin call', 0.4808, 101.0, 101.0),
        ('A', 'Close entry(s) order A', None, 1.5, 100.0, 101.0),
        ('A', 'Close position order', None, 13.0192, 100.0, 101.0),
        ('B', 'Close position order', None, 4.4192, 100.0, 101.0),
        ('C', 'Close position order', None, 0.5192, 101.0, 101.0),
        ('D', 'Close position order', None, 1.0, 101.0, 101.0),
    ], shape
