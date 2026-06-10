"""
@pyne

Regression test for the long trailing stop riding the bar intrabar.

TradingView's broker emulator moves the price along the assumed intrabar path
(open -> low -> high -> close here) and the trailing stop follows it tick by
tick: on the activation bar the high-water mark advances to the bar's own high
and the stop fills on the pullback at ``high - trail_offset`` — NOT at the
activation level. Verified against TradingView references on BINANCE:BTCUSDT
(e.g. 2025-01-15 19:30: high 99990, offset ~10 -> TV fill 99980; PyneCore
previously filled at activation - offset, 99943.6).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Intrabar Ride Long",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    if bar_index == 0 and strategy.position_size == 0:
        strategy.entry('L', strategy.long)
    if strategy.position_size > 0:
        strategy.exit('X', 'L', trail_price=101.00, trail_offset=10)


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
def __test_trailing_ride_fills_at_high_minus_offset__(script_path, module_key):
    """
    Same-bar activation, intrabar ride to the high, pullback fill.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (trail_price=101.00, offset=10 ticks
      = 0.10); the close (100.10) is below the activation level, so no arming.
    * bar 2: open 100.50 -> low 100.40 -> high 102.00 -> close 101.50. The trail
      arms at 101.00 on the rising leg, the water mark rides to the bar high
      102.00, and the closing leg pulls back through 101.90 = high - offset.
      The fill must land at 101.90 on this same bar.

    The pre-fix behavior anchored the stop at the activation level
    (101.00 - 0.10 = 100.90) and filled there.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - entry signal
        (100.00, 100.20, 100.00, 100.10),  # bar 1 - entry fill, exit issued
        (100.50, 102.00, 100.40, 101.50),  # bar 2 - arm, ride to 102.00, fill
        (101.50, 101.70, 101.40, 101.60),  # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, f"expected exactly one closed trade, got {len(trades)}"
    t = trades[0]
    assert t.entry_bar_index == 1, f"entry_bar_index={t.entry_bar_index}"
    assert abs(t.entry_price - 100.00) < 1e-9, f"entry_price={t.entry_price}"
    assert t.exit_bar_index == 2, (
        f"exit should land on the activation bar 2, got {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 101.90) < 1e-9, (
        f"trail should ride to the bar high and fill at 102.00 - 0.10 = 101.90, "
        f"got {t.exit_price}"
    )
