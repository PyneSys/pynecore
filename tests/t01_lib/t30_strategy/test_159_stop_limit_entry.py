"""
@pyne

A ``strategy.entry`` with both ``stop`` and ``limit`` is ONE stop-limit order.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, probes sl1-sl8): the order rests as a stop;
its trigger turns it into a limit order that fills at the trigger price when marketable
there, else later at its limit -- on the same bar's way back too. Re-issued with the same
prices it keeps its triggered state, with changed prices it is a fresh stop again. A
default-sized order is sized at ``min(max(stop, close), limit)`` of the placement bar
wherever it fills.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Stop-Limit Entry",
    overlay=True,
    initial_capital=10000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=10,
)
def main():
    # sized at min(max(101, 100), 105) = 101, gap-filled at the 103 open
    if bar_index == 0:
        strategy.entry('S', strategy.long, stop=101.0, limit=105.0)
    # the trigger at 101 is marketable for a 102 limit
    if bar_index == 3:
        strategy.entry('A', strategy.long, qty=1, stop=101.0, limit=102.0)
    # triggered at the high, filled at its limit on the way back to the open
    if bar_index == 6:
        strategy.entry('B', strategy.long, qty=1, stop=101.0, limit=100.5)
    # triggered on bar 10, re-issued unchanged, filled at its limit on bar 12
    if 9 <= bar_index <= 11 and strategy.position_size == 0:
        strategy.entry('C', strategy.long, qty=1, stop=101.0, limit=100.5)
    # triggered on bar 16, re-issued with a new limit: a fresh stop, never filled
    if bar_index == 15:
        strategy.entry('D', strategy.long, qty=1, stop=101.0, limit=100.5)
    if bar_index == 16:
        strategy.entry('D', strategy.long, qty=1, stop=101.0, limit=100.55)
    if bar_index == 18:
        strategy.cancel('D')
    if bar_index == 1 or bar_index == 4 or bar_index == 7 or bar_index == 13:
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
    return [(t.entry_id, t.entry_bar_index, round(abs(t.size), 6), round(t.entry_price, 2),
             round(t.exit_price, 2)) for t in trades]


# noinspection PyShadowingNames
def __test_stop_limit_entry__(script_path, module_key):
    """Stop trigger, limit fill, re-issue semantics and budget sizing of a stop-limit entry"""
    flat = (100.0, 100.05, 99.95, 100.0)
    rows = [flat] * 20
    rows[1] = (103.0, 104.0, 102.5, 103.5)
    rows[4] = (100.0, 101.5, 99.95, 101.2)
    rows[7] = (100.0, 101.2, 99.0, 99.5)
    rows[10] = rows[16] = (100.0, 101.2, 100.6, 101.0)
    rows[11] = (101.0, 101.05, 100.7, 100.8)
    rows[12] = rows[17] = (100.8, 100.85, 100.4, 100.6)
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('S', 1, 9.9009, 103.0, 100.0),
        ('A', 4, 1.0, 101.0, 100.0),
        ('B', 7, 1.0, 100.5, 100.0),
        ('C', 12, 1.0, 100.5, 100.0),
    ], shape
