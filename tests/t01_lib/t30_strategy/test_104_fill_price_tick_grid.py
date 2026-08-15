"""
@pyne

Regression test: fills are booked on the tick grid and the average entry price is
the accumulated cost divided by the position size.

Both are single-ULP effects, and both are measured: on BINANCE:BTCUSDT@30 the
entry price of 3408 of 3408 open trades is ``round(price / mintick) * mintick``,
and ``strategy.position_avg_price`` equals ``sum(entry_price * size) / sum(size)``
on all 22720 in-position bars -- an incremental re-weighting of the previous
average, which is algebraically the same, misses 9280 of them.
"""
from pynecore.lib import script, strategy, plot, bar_index


@script.strategy(
    "Fill Price Tick Grid",
    initial_capital=1_000_000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=10,
    process_orders_on_close=True,
)
def main():
    if bar_index == 0:
        strategy.entry('A', strategy.long, qty=0.001)
    if bar_index == 1:
        strategy.entry('B', strategy.long, qty=0.003)

    plot(strategy.position_avg_price, "avg")
    plot(strategy.opentrades.entry_price(0), "e0")
    plot(strategy.opentrades.entry_price(1), "e1")


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='30', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_fill_price_and_average_follow_tradingview__(script_path, module_key):
    """
    Two entries at the bar close, on the exact prices the probe pyramided on.

    * bar 0 closes at 93761.9, which is NOT on the tick grid as a double: the
      booked price is 93761.90000000001.
    * bar 1 closes at 94098.91, which already is on the grid.
    * the resulting average is 94014.65750000002 -- the cost sum divided by 0.004.
      Re-weighting the first average by 0.003/0.004 gives 94014.6575, an ULP below.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    closes = [93761.9, 94098.91, 93830.89, 93838.04]
    bars = [
        OHLCV(timestamp=base_ts + i * 1_800_000, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    plots = [dict(p) for _candle, p, _closed in runner.run_iter()]

    # The fills land on the bar close, so the state shows up on the next bar.
    assert plots[1]["e0"] == 93761.90000000001, plots[1]["e0"]
    assert plots[1]["avg"] == 93761.90000000001, plots[1]["avg"]

    assert plots[2]["e0"] == 93761.90000000001, plots[2]["e0"]
    assert plots[2]["e1"] == 94098.91, plots[2]["e1"]
    assert plots[2]["avg"] == 94014.65750000002, plots[2]["avg"]
    assert plots[3]["avg"] == 94014.65750000002, plots[3]["avg"]
