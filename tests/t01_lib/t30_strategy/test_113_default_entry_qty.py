"""
@pyne

Regression test: ``strategy.default_entry_qty()`` snaps its price argument onto the
tick grid, then floors the resulting size onto the lot grid.

Measured on TradingView (BINANCE:BTCUSDT 30m, mintick 0.01, lot step 1e-5,
initial_capital 13337, percent_of_equity 17.3, no commission): 100.007 and 100.01
both size 23.0707 while 100.003 and 100.0 both size 23.07301, 0.005 sizes off 0.01
where 0.004 snaps to zero, and a price that snaps to zero -- like an na price --
returns 0 instead of erroring. A negative price keeps its sign and truncates
toward zero.
"""
from pynecore.lib import script, strategy, plot, close, na


@script.strategy(
    "Default Entry Qty",
    initial_capital=13337,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=17.3,
)
def main():
    plot(strategy.default_entry_qty(close), "at_close")
    plot(strategy.default_entry_qty(-close), "negative")
    plot(strategy.default_entry_qty(na(float)), "na_price")
    plot(strategy.default_entry_qty(0.0), "zero_price")
    plot(strategy.default_entry_qty(100.0), "p100")
    plot(strategy.default_entry_qty(100.003), "p100_003")
    plot(strategy.default_entry_qty(100.007), "p100_007")
    plot(strategy.default_entry_qty(100.01), "p100_01")
    plot(strategy.default_entry_qty(0.004), "p0_004")
    plot(strategy.default_entry_qty(0.006), "p0_006")


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='30', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.00001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_default_entry_qty_follows_tradingview__(script_path, module_key):
    """
    The script never trades, so equity stays at the initial capital and every bar
    reports the same sizes TradingView plotted for the same configuration.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC, in ms
    closes = [93761.9, 94401.14, 93825.86]
    bars = [
        OHLCV(timestamp=base_ts + i * 1_800_000, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    plots = [dict(p) for _candle, p, _closed in runner.run_iter()]

    # Price-driven sizes, floored on the lot grid.
    assert [p["at_close"] for p in plots] == [0.0246, 0.02444, 0.02459]
    assert [p["negative"] for p in plots] == [-0.0246, -0.02444, -0.02459]

    # Nothing to size from: na, and a price snapping to zero.
    assert [p["na_price"] for p in plots] == [0.0, 0.0, 0.0]
    assert [p["zero_price"] for p in plots] == [0.0, 0.0, 0.0]
    assert [p["p0_004"] for p in plots] == [0.0, 0.0, 0.0]

    # The price is snapped to the tick grid before the division, so an off-grid
    # price sizes exactly like the tick it rounds to.
    for p in plots:
        assert p["p100"] == 23.07301, p["p100"]
        assert p["p100_003"] == p["p100"], p["p100_003"]
        assert p["p100_01"] == 23.070700000000002, p["p100_01"]
        assert p["p100_007"] == p["p100_01"], p["p100_007"]
        assert p["p0_006"] == 230730.09999000002, p["p0_006"]
