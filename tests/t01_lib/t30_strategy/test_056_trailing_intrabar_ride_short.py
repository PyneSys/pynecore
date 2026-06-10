"""
@pyne

Regression test for the short trailing stop riding the bar intrabar.

Mirror of the long case: on the activation bar the low-water mark advances to
the bar's own low along the assumed intrabar path (open -> high -> low -> close
here) and the buy-stop fills on the bounce at ``low + trail_offset`` — NOT at
the activation level. The long side is verified against TradingView references
on BINANCE:BTCUSDT (short trailing fills there show the same
extreme-plus-offset law, e.g. 2026-02-05 15:00/15:30).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Intrabar Ride Short",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    if bar_index == 0 and strategy.position_size == 0:
        strategy.entry('S', strategy.short)
    if strategy.position_size < 0:
        strategy.exit('X', 'S', trail_price=99.00, trail_offset=10)


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
def __test_trailing_ride_fills_at_low_plus_offset__(script_path, module_key):
    """
    Same-bar activation, intrabar ride to the low, bounce fill.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (trail_price=99.00, offset=10 ticks
      = 0.10); the close (99.90) is above the activation level, so no arming.
    * bar 2: open 99.50 -> high 99.60 -> low 98.00 -> close 98.50. The trail
      arms at 99.00 on the falling leg, the water mark rides to the bar low
      98.00, and the closing leg bounces through 98.10 = low + offset. The fill
      must land at 98.10 on this same bar.

    The pre-fix behavior anchored the stop at the activation level
    (99.00 + 0.10 = 99.10) and filled there.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    rows = [
        # open,  high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - entry signal
        (100.00, 100.00, 99.80, 99.90),    # bar 1 - entry fill, exit issued
        (99.50, 99.60, 98.00, 98.50),      # bar 2 - arm, ride to 98.00, fill
        (98.50, 98.60, 98.30, 98.40),      # bar 3 - tail
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
    assert abs(t.exit_price - 98.10) < 1e-9, (
        f"trail should ride to the bar low and fill at 98.00 + 0.10 = 98.10, "
        f"got {t.exit_price}"
    )
