"""
@pyne

A price-based ``strategy.entry`` is judged against the pyramiding limit only when it is
placed, never when it fills.

MEASURED on TradingView (CAPITALCOM:EURUSD 30m, ``pyramiding=1``, probe "PYR fill probe",
1191 cycles): of two long stop entries placed together while flat, the second fills
whenever it triggers -- on the first one's fill bar (376/376) and on a later bar with the
first one already open (529/529). Repeating the second entry's call once the first one is
open rejects that call, and the rejected call removes the resting order: it never fills
afterwards (0/529).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Price Entry Pyramiding At Placement",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    # A and B trigger on the same bar
    if bar_index == 0:
        strategy.entry('A', strategy.long, stop=100.5)
        strategy.entry('B', strategy.long, stop=101.0)
    if bar_index == 2:
        strategy.close_all()

    # D triggers two bars after C has opened the position
    if bar_index == 5:
        strategy.entry('C', strategy.long, stop=100.5)
        strategy.entry('D', strategy.long, stop=102.0)
    if bar_index == 10:
        strategy.close_all()

    # F is re-issued while E is open: the rejected call removes it
    if bar_index == 13:
        strategy.entry('E', strategy.long, stop=100.5)
        strategy.entry('F', strategy.long, stop=102.0)
    if bar_index == 14:
        strategy.entry('F', strategy.long, stop=102.0)
    if bar_index == 18:
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


# noinspection PyShadowingNames
def __test_price_entry_pyramiding_at_placement__(script_path, module_key):
    """Both stops placed while flat fill; a re-issued one is removed by its rejected call"""
    flat = (100.0, 100.05, 99.95, 100.0)
    rows = [flat] * 22
    rows[1] = (100.0, 101.5, 99.95, 101.0)
    rows[6] = rows[14] = (100.0, 100.6, 99.95, 100.2)
    rows[8] = rows[16] = (100.0, 102.5, 99.95, 100.2)
    trades = __test_helper_run(script_path, module_key, rows)

    shape = [(t.entry_id, t.entry_bar_index, round(t.entry_price, 2), t.exit_bar_index)
             for t in trades]
    assert shape == [
        ('A', 1, 100.5, 3),
        ('B', 1, 101.0, 3),
        ('C', 6, 100.5, 11),
        ('D', 8, 102.0, 11),
        ('E', 14, 100.5, 19),
    ], shape
