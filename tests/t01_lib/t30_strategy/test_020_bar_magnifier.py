"""
@pyne

Bar magnifier end-to-end test.

This strategy enters long on bar 1, then places a take-profit (limit sell) at 110
and a stop-loss (stop sell) at 90. The test creates synthetic 1-minute data where
the price path within the chart bar is KNOWN — we control whether price goes up
first or down first. This lets us verify that the magnifier processes fills in the
correct chronological order, not based on the OHLC direction heuristic.
"""
from pynecore.lib import close, script, strategy

# Simple strategy: enter long on first bar, then TP/SL
@script.strategy(
    "Bar Magnifier Test",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    use_bar_magnifier=True,
)
def main():
    if strategy.position_size == 0:
        strategy.entry('Long', strategy.long)

    if strategy.position_size > 0:
        strategy.exit(
            'TP/SL', from_entry='Long',
            limit=110.0,  # Take profit at 110
            stop=90.0,    # Stop loss at 90
        )


def _make_sub_bars(base_ts: int, prices: list[tuple[float, float, float, float]]) -> list:
    """
    Create 1-minute OHLCV sub-bars from a list of (open, high, low, close) tuples.

    :param base_ts: Starting timestamp in seconds
    :param prices: List of (O, H, L, C) tuples, one per minute
    :return: List of OHLCV objects
    """
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(prices)
    ]


def _make_syminfo(period: str = '5'):
    """Create a minimal SymInfo for testing."""
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider.get_opening_hours_and_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_magnifier_tp_hits_before_sl_when_price_goes_up_first__(script_path, module_key):
    """
    Bar magnifier: take-profit fills when price goes UP first within the bar.

    Chart bar: O=100, H=112, L=88, C=100
    Without magnifier: heuristic might guess O→L→H→C (down first), hitting SL at 90.
    With magnifier: actual 1-min path goes UP first to 112, so TP at 110 fills first.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='5')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    # Bar 0: entry bar — flat, just to get a position opened
    # (market order fills at open of next bar, so we need bar 0 for the entry signal
    #  and bar 1 for the fill + exit processing)
    bar0_subs = _make_sub_bars(base_ts, [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ])

    # Bar 1: market order fills at open (100), then exit orders are placed.
    # But exits won't process until bar 2.
    bar1_subs = _make_sub_bars(base_ts + 300, [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ])

    # Bar 2: price goes UP FIRST then down — TP at 110 should hit before SL at 90
    bar2_subs = _make_sub_bars(base_ts + 600, [
        (100.0, 105.0, 100.0, 105.0),  # up
        (105.0, 112.0, 105.0, 110.0),  # up more, hits TP at 110
        (110.0, 110.0, 95.0,   95.0),  # drops (but TP already filled!)
        (95.0,   95.0, 88.0,   90.0),  # drops more (would hit SL, but already closed)
        (90.0,   92.0, 88.0,   92.0),  # recovery
    ])

    all_sub_bars = bar0_subs + bar1_subs + bar2_subs

    runner = ScriptRunner(
        Path(script_path), iter([]), syminfo,
        magnifier_iter=iter(all_sub_bars),
    )

    trades = []
    for candle, plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    # Should have exactly 1 closed trade
    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

    trade = trades[0]
    # Entry at 100 (market order at bar 1 open)
    assert trade.entry_price == 100.0, f"Expected entry at 100, got {trade.entry_price}"
    # TP at 110 should have filled (price went up first!)
    assert trade.exit_price == 110.0, f"Expected TP exit at 110, got {trade.exit_price}"
    # Profit should be positive (long: 110 - 100 = 10)
    assert trade.profit > 0, f"Expected positive profit, got {trade.profit}"


# noinspection PyShadowingNames
def __test_magnifier_sl_hits_before_tp_when_price_goes_down_first__(script_path, module_key):
    """
    Bar magnifier: stop-loss fills when price goes DOWN first within the bar.

    Same chart bar OHLC as above (O=100, H=112, L=88, C=100), but the 1-minute
    path goes DOWN first. So SL at 90 fills before TP at 110 — opposite result
    from the same aggregated bar, which is exactly why the magnifier exists.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='5')
    base_ts = 1704067200

    # Bar 0: entry bar
    bar0_subs = _make_sub_bars(base_ts, [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ])

    # Bar 1: fill bar
    bar1_subs = _make_sub_bars(base_ts + 300, [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ])

    # Bar 2: price goes DOWN FIRST then up — SL at 90 should hit before TP at 110
    bar2_subs = _make_sub_bars(base_ts + 600, [
        (100.0, 100.0, 95.0,  95.0),   # down
        (95.0,  95.0,  88.0,  90.0),   # down more, hits SL at 90
        (90.0,  105.0, 90.0,  105.0),  # recovers (but SL already filled!)
        (105.0, 112.0, 105.0, 110.0),  # up more (would hit TP, but already closed)
        (110.0, 112.0, 108.0, 108.0),  # settles
    ])

    all_sub_bars = bar0_subs + bar1_subs + bar2_subs

    runner = ScriptRunner(
        Path(script_path), iter([]), syminfo,
        magnifier_iter=iter(all_sub_bars),
    )

    trades = []
    for candle, plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

    trade = trades[0]
    assert trade.entry_price == 100.0, f"Expected entry at 100, got {trade.entry_price}"
    # SL at 90 should have filled (price went down first!)
    assert trade.exit_price == 90.0, f"Expected SL exit at 90, got {trade.exit_price}"
    # Profit should be negative (long: 90 - 100 = -10)
    assert trade.profit < 0, f"Expected negative profit, got {trade.profit}"
