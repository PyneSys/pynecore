"""
@pyne

A sub-lot shortfall at the fill price liquidates nothing on the leg; the bar-close
check sizes the cut in lots.

MEASURED on the wild `Turtle Trader Strategy` (BINANCE:BTCUSDT 30m, 2025-01-19 09:00):
a sell stop filling 0.85 lot short of margin is not touched at the low; the bar close
trims 4x the lots of the shortfall there (8 lots), not one whole contract.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Margin Call Sub Lot At Fill Price",
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
        strategy.order('S', strategy.short, qty=10.0232, stop=99.5)
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
def __test_sub_lot_fill_shortfall_waits_for_the_close__(script_path, module_key):
    """
    At the 99.5 fill the 10.0232 short is 0.0084 (0.84 lot) short of 997.3; at the
    99.6 close it is 2.013 short = 202 lots, cut 4x = 0.0808 at the close.
    """
    flat = (100.0, 100.05, 99.95, 100.0)
    calm = (99.5, 99.55, 99.45, 99.5)
    rows = [flat, flat, (100.0, 100.05, 98.5, 99.6), calm, calm, calm, calm]
    trades = __test_helper_run(script_path, module_key, rows)

    shape = __test_helper_shape(trades)
    assert shape == [
        ('L', 'X', None, 9.0, 100.0, 99.7),
        ('S', None, 'Margin call', 0.0808, 99.5, 99.6),
        ('S', 'Close position order', None, 9.9424, 99.5, 99.5),
    ], shape
