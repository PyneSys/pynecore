"""
@pyne

strategy.margin_liquidation_price with a zero margin percent on the position's side.

Measured on TradingView (BINANCE:BTCUSDT 1D, margin_long=0, margin_short=30):
an open long reports na — the side that requires no margin can never be margin
called — while the short side still reports its liquidation price.
"""
from pynecore.lib import bar_index, plot, script, strategy


@script.strategy(
    "Margin liquidation price no margin",
    overlay=True,
    initial_capital=100000,
    margin_short=30,
)
def main():
    if bar_index == 1:
        strategy.entry('L', strategy.long, qty=2)
    if bar_index == 5:
        strategy.close_all()
    if bar_index == 7:
        strategy.entry('S', strategy.short, qty=4)

    plot(strategy.margin_liquidation_price, "mlp")
    plot(strategy.position_size, "size")


def _make_syminfo():
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
def __test_no_margin_side_reports_na__(script_path, module_key):
    """margin_long=0 makes an open long report na; the short side still reports a price."""
    import sys
    import math
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=100.0 + 10 * i, high=110.0 + 10 * i,
              low=95.0 + 10 * i, close=105.0 + 10 * i, volume=100.0)
        for i in range(10)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    rows = [dict(plot_values) for _candle, plot_values, _closed in runner.run_iter()]

    for i in (2, 3, 4, 5):
        assert rows[i]['size'] == 2.0, f"bar {i}: long not open: {rows[i]['size']}"
        assert math.isnan(rows[i]['mlp']), f"bar {i}: unfunded long is not na: {rows[i]['mlp']}"

    # Same trade sequence as test_086, so the short value is the same 19384.62
    for i in (8, 9):
        assert rows[i]['size'] == -4.0, f"bar {i}: short not open: {rows[i]['size']}"
        assert abs(rows[i]['mlp'] - 19384.62) < 1e-9, f"bar {i}: {rows[i]['mlp']}"
