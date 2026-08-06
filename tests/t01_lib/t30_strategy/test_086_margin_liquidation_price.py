"""
@pyne

strategy.margin_liquidation_price — the price where the margin call occurs.

Measured on TradingView (BINANCE:BTCUSDT 1D and FX:EURUSD 1D, margin_long=25,
margin_short=30): the value solves ``equity(P) = margin(P)`` — the same balance
the margin-call check compares — and snaps to the tick grid directionally:
a long floors toward -inf (a negative price stays reportable), a short ceils.
Flat bars report na.
"""
from pynecore.lib import bar_index, plot, script, strategy


@script.strategy(
    "Margin liquidation price",
    overlay=True,
    initial_capital=100000,
    margin_long=25,
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
def __test_margin_liquidation_price_solves_the_margin_balance__(script_path, module_key):
    """Long floors, short ceils the equity/margin break-even price; flat bars are na."""
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

    assert len(rows) == len(bars), "the run must reach the last bar"

    # Long: fills at bar-2 open = 120, qty 2, margin 25%:
    # P = (2*120 - 100000) / (2 * 0.75) = -66506.666..., floored to -66506.67 —
    # the floor (not trunc) of a NEGATIVE price is the measured TV behaviour
    for i in (2, 3, 4, 5):
        assert rows[i]['size'] == 2.0, f"bar {i}: long not open: {rows[i]['size']}"
        assert abs(rows[i]['mlp'] - (-66506.67)) < 1e-9, f"bar {i}: {rows[i]['mlp']}"

    # Short: closed at bar-6 open = 160 (netprofit 80), fills at bar-8 open = 180,
    # qty 4, margin 30%: P = (100080 + 4*180) / (4 * 1.3) = 19384.615..., ceiled
    for i in (8, 9):
        assert rows[i]['size'] == -4.0, f"bar {i}: short not open: {rows[i]['size']}"
        assert abs(rows[i]['mlp'] - 19384.62) < 1e-9, f"bar {i}: {rows[i]['mlp']}"

    # Flat bars (before the first fill and the bar the close fills on) are na
    for i in (0, 1, 6, 7):
        assert rows[i]['size'] == 0.0, f"bar {i}: not flat: {rows[i]['size']}"
        assert math.isnan(rows[i]['mlp']), f"bar {i}: flat mlp is not na: {rows[i]['mlp']}"
