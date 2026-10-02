"""
@pyne

Under ``close_entries_rule='ANY'`` a batch's exit leg whose own trades are already
closed runs after the legs that still have trades to close.

MEASURED on the wild `Turtle Trader Strategy` (BINANCE:BTCUSDT 30m, 6/6 events): a
``strategy.order`` buy and an id-less ``strategy.exit`` stop share one level; the buy
closes the first short entry and part of a pyramid add, the add's leg closes the rest
of the add and opens the part the buy took, then the spent entry's leg opens its whole
quantity -- every opened part under the exit id.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "ANY Batch Spent Leg Runs Last",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=2,
    close_entries_rule="ANY",
)
def main():
    if bar_index == 0:
        strategy.entry('S2', strategy.short, qty=5)
    if bar_index == 1:
        strategy.entry('P', strategy.short, qty=2)
    if bar_index == 2:
        strategy.order('LO', strategy.long, qty=6, stop=101)
        strategy.exit('X', stop=101)
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
def __test_spent_leg_opens_after_live_leg__(script_path, module_key):
    """The buy closes S2 (5) and 1 of P; P's leg closes P's last 1 and opens 1, S2's opens 5"""
    flat = (100.0, 100.05, 99.95, 100.0)
    rows = [flat, flat, flat, (100.0, 101.5, 99.9, 101.0), flat, flat, flat]
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('S2', 'LO', None, 5.0, 100.0, 101.0),
        ('P', 'LO', None, 1.0, 100.0, 101.0),
        ('P', 'X', None, 1.0, 100.0, 101.0),
        ('X', 'Close position order', None, 1.0, 101.0, 100.0),
        ('X', 'Close position order', None, 5.0, 101.0, 100.0),
    ], shape
