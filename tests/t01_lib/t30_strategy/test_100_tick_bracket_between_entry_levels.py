"""
@pyne

Regression test for a tick-based bracket sitting between two pending entry levels.

Bar 0 places a long stop entry at 100.00 with a `strategy.exit(profit=50)`
take-profit (100.50 once the entry fills) and a second long stop entry at 101.00.
Bar 1's rising leg reaches all three in price order, so TradingView closes the first
trade at 100.50 and the position is flat again when 101.00 is reached — the second
entry fills instead of being rejected by the pyramiding limit.

The bracket only gains a price level when its entry fills, i.e. in the middle of the
leg. A walk that appends such a late-materialized level after the whole leg reaches
101.00 while the first trade is still open, and the second entry is dropped.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Tick Bracket Between Entry Levels",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L1', strategy.long, stop=100.00)
        strategy.exit('L1X', from_entry='L1', profit=50)  # ticks -> 100.00 + 0.50
        strategy.entry('L2', strategy.long, stop=101.00)
    elif bar_index == 3:
        strategy.close_all()


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_bracket_frees_the_pyramiding_slot_before_the_next_level__(script_path, module_key):
    """
    A bracket reached mid-leg closes its trade before a later entry level is reached.

    * bar 0: both stop entries and the 50-tick take-profit are placed.
    * bar 1: O=99.90 H=101.50 L=97.00 C=99.00 -> path open -> high -> low -> close.
      The ascent fills L1 at 100.00, its take-profit at 100.50 and L2 at 101.00.
    * bar 3 closes whatever is still open, so L2's fill is visible as a closed trade.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms

    rows = [
        # open,   high,   low,    close
        (98.00, 98.50, 97.50, 98.00),     # bar 0 - orders placed, no level reached
        (99.90, 101.50, 97.00, 99.00),    # bar 1 - L1 @100.00, TP @100.50, L2 @101.00
        (99.00, 99.20, 98.80, 99.00),     # bar 2
        (99.00, 99.20, 98.80, 99.00),     # bar 3 - close_all placed
        (99.00, 99.20, 98.80, 99.00),     # bar 4 - close_all fills at the open
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, (
        f"Expected 2 closed trades, got {len(trades)} — the take-profit has to free the "
        f"pyramiding slot before the walk reaches the second entry level"
    )

    first, second = trades
    assert abs(first.entry_price - 100.0) < 1e-9, f"first entry price {first.entry_price}"
    assert first.exit_bar_index == 1, f"first exit bar {first.exit_bar_index}"
    assert abs(first.exit_price - 100.5) < 1e-9, f"first exit price {first.exit_price}"

    assert second.entry_bar_index == 1, f"second entry bar {second.entry_bar_index}"
    assert abs(second.entry_price - 101.0) < 1e-9, f"second entry price {second.entry_price}"
