"""
@pyne

A default-sized stop entry the open gaps through is sized at the price it would
have executed at when it was placed — its stop level, or the placement close
when that was already past the level — pushed by the slippage, and NOT at the
open it actually fills at.

Measured on TradingView (CAPITALCOM:BTCUSD 60, slippage 3, ~7800 trades per
probe), identically for strategy.entry and strategy.order: a buy stop one tick
above the close sized every gapped fill at stop + slippage; a buy stop below
the close sized every one at close + slippage.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Gapped Stop Entry Sizing",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.cash,
    default_qty_value=200000,
    slippage=3,
)
def main():
    # bar 0: buy stop BELOW the 110 close -> already past its level
    if bar_index == 0:
        strategy.entry("A", strategy.long, stop=105.0)
    if bar_index == 2:
        strategy.close("A")
    # bar 4: buy stop ABOVE the 110 close -> rests at its level
    if bar_index == 4:
        strategy.entry("B", strategy.long, stop=112.0)
    if bar_index == 6:
        strategy.close("B")
    # bars 8 and 12: the same two stops placed through strategy.order
    if bar_index == 8:
        strategy.order("C", strategy.long, stop=105.0)
    if bar_index == 10:
        strategy.close("C")
    if bar_index == 12:
        strategy.order("D", strategy.long, stop=112.0)
    if bar_index == 14:
        strategy.close("D")


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


# noinspection PyShadowingNames
def __test_gapped_stop_entry_sized_at_placement_price__(script_path, module_key):
    """Every stop fills at the 115 open, sized at close+slip or stop+slip."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    # One 4-bar cycle per stop: placed on a 110 close, gapped by the 115 open
    # of the next bar, closed on the bar after that.
    cycle = [
        # open,   high,   low,    close
        (110.00, 110.00, 110.00, 110.00),  # stop placed
        (115.00, 116.00, 114.00, 115.00),  # gapped: fills at the open
        (115.00, 115.00, 115.00, 115.00),  # close the trade
        (110.00, 110.00, 110.00, 110.00),
    ]
    rows = cycle * 4
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert [t.entry_id for t in trades] == ["A", "B", "C", "D"], trades
    slip = 0.03
    sizing_prices = (110.00 + slip, 112.00 + slip, 110.00 + slip, 112.00 + slip)
    for trade, sizing_price in zip(trades, sizing_prices):
        assert abs(trade.entry_price - (115.00 + slip)) < 1e-9, \
            f"{trade.entry_id}: entry_price={trade.entry_price}"
        expected = 200000.0 / sizing_price
        assert abs(trade.size - expected) < 1e-4, (
            f"{trade.entry_id}: size={trade.size}, expected ~{expected} "
            f"(the fill price would give {200000.0 / trade.entry_price})"
        )
