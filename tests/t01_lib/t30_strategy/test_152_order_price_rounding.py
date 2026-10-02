"""
@pyne

``strategy.order`` rounds its price levels to the tick grid like ``strategy.entry``.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, off-grid levels of all four kinds,
2644/2644 fills): a buy limit and a sell stop round down, a sell limit and a buy stop
round up.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Order Price Rounding",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.order('BuyLimit', strategy.long, qty=1, limit=99.987)
    if bar_index == 3:
        strategy.order('SellLimit', strategy.short, qty=1, limit=100.013)
    if bar_index == 6:
        strategy.order('BuyStop', strategy.long, qty=1, stop=100.013)
    if bar_index == 9:
        strategy.order('SellStop', strategy.short, qty=1, stop=99.987)
    if bar_index == 2 or bar_index == 5 or bar_index == 8 or bar_index == 11:
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
def __test_order_levels_round_like_entry__(script_path, module_key):
    """Each order fills on the tick its kind rounds to, inside an un-gapped bar"""
    rows = [(100.0, 100.1, 99.9, 100.0)] * 13
    trades = __test_helper_run(script_path, module_key, rows)

    entries = [(t.entry_id, round(t.entry_price, 2)) for t in trades]
    assert entries == [('BuyLimit', 99.98), ('SellLimit', 100.02), ('BuyStop', 100.02),
                       ('SellStop', 99.98)], entries
