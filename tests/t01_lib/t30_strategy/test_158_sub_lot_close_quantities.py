"""
@pyne

Close quantities below one lot.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, minlot probes): a ``qty_percent`` share
below one lot still closes ONE lot, in ``strategy.close`` and in ``strategy.exit`` legs
alike; an explicit ``qty`` that floors to zero lots (``0`` or half a lot) acts like an
omitted one -- ``strategy.close`` closes the whole entry, a ``strategy.exit`` leg becomes
a rest leg; a runtime negative ``strategy.close`` qty closes its absolute value.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Sub-Lot Close Quantities",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    # 19 lots, a 5% close: 0.95 lot -> one lot
    if bar_index == 0:
        strategy.entry('A', strategy.long, qty=0.0019)
    if bar_index == 1:
        strategy.close('A', qty_percent=5)
    if bar_index == 2:
        strategy.close_all()
    # half a lot -> the whole entry
    if bar_index == 3:
        strategy.entry('B', strategy.long, qty=0.005)
    if bar_index == 4:
        strategy.close('B', qty=0.00005)
    # zero -> the whole entry
    if bar_index == 5:
        strategy.entry('C', strategy.long, qty=0.005)
    if bar_index == 6:
        strategy.close('C', qty=0.0)
    # runtime negative -> its absolute value
    if bar_index == 7:
        strategy.entry('D', strategy.long, qty=0.005)
    if bar_index == 8:
        strategy.close('D', qty=-0.001 if bar_index > 0 else 0.001)
    if bar_index == 9:
        strategy.close_all()
    # a half-lot exit leg beside a 20-lot sibling closes the other 30 lots
    if bar_index == 10:
        strategy.entry('E', strategy.long, qty=0.005)
    if bar_index == 11:
        strategy.exit('E1', 'E', qty=0.002, limit=200.0)
        strategy.exit('E2', 'E', qty=0.00005, limit=101.0)
    if bar_index == 13:
        strategy.close_all()
    # two 1% legs of 50 lots: one lot each
    if bar_index == 14:
        strategy.entry('F', strategy.long, qty=0.005)
    if bar_index == 15:
        strategy.exit('F1', 'F', qty_percent=1, limit=101.0)
        strategy.exit('F2', 'F', qty_percent=1, limit=101.0)
    if bar_index == 16:
        strategy.close_all()
    # siblings reserving 30% + 70% leave nothing: a third leg gets no lot
    if bar_index == 17:
        strategy.entry('G', strategy.long, qty=0.001)
    if bar_index == 18:
        strategy.exit('G1', 'G', qty_percent=30, limit=200.0)
        strategy.exit('G2', 'G', qty_percent=70, limit=200.0)
        strategy.exit('G3', 'G', qty_percent=50, limit=101.0)
    if bar_index == 19:
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
    return [(t.entry_id, t.exit_id, round(abs(t.size), 6), round(t.entry_price, 2),
             round(t.exit_price, 2)) for t in trades]


# noinspection PyShadowingNames
def __test_sub_lot_close_quantities__(script_path, module_key):
    """qty_percent keeps one lot, a zero-lot qty closes like no qty, a negative qty its abs"""
    flat = (100.0, 100.05, 99.95, 100.0)
    up = (100.0, 101.5, 99.95, 100.0)
    rows = [flat] * 21
    rows[12] = rows[16] = rows[19] = up
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('A', 'Close entry(s) order A', 0.0001, 100.0, 100.0),
        ('A', 'Close position order', 0.0018, 100.0, 100.0),
        ('B', 'Close entry(s) order B', 0.005, 100.0, 100.0),
        ('C', 'Close entry(s) order C', 0.005, 100.0, 100.0),
        ('D', 'Close entry(s) order D', 0.001, 100.0, 100.0),
        ('D', 'Close position order', 0.004, 100.0, 100.0),
        ('E', 'E2', 0.003, 100.0, 101.0),
        ('E', 'Close position order', 0.002, 100.0, 100.0),
        ('F', 'F1', 0.0001, 100.0, 101.0),
        ('F', 'F2', 0.0001, 100.0, 101.0),
        ('F', 'Close position order', 0.0048, 100.0, 100.0),
        ('G', 'Close position order', 0.001, 100.0, 100.0),
    ], shape
