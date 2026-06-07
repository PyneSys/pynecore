"""
@pyne

Regression test for the carried long trailing-stop fill on a gap-up open bar.

A long position's trailing stop (``trail_offset=0``) rides the high-water mark.
When the NEXT bar opens at its own low (no lower wick) and that open gaps above
the carried water mark, TradingView folds the open into the high-water mark and
the trail fills at the open -- the bar is already at its lowest there and never
trades back down. PyneCore previously kept the prior bar's water mark, so the
open (one tick above it) never reached the stop and the trail rode on, diverging
from TradingView by a full tick (and, against a competing limit, by a whole bar).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Gap-Open Long",
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
        strategy.exit('X', 'L', trail_points=5, trail_offset=0)


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='1', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_gap_open_long_fills_at_open__(script_path, module_key):
    """
    The carried long trail fills at the gap-up open of a no-lower-wick bar.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: rises to a high of 100.20 and closes there, arming the trail and
      ratcheting the offset-0 water mark to 100.20 (no same-bar fill: the close
      is the high, so the bar never retraces below it).
    * bar 2: opens at its own low 100.21 -- one tick above the carried water
      mark -- then runs up. The trail must fill at the open, 100.21.

    Without the open-water-mark fold the stop stays at 100.20, the bar's low
    (100.21) never reaches it, and the trail rides up to 100.60 instead.
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
        (100.00, 100.20, 100.00, 100.20),  # bar 1 - entry fill, trail arms, wm=100.20
        (100.21, 100.60, 100.21, 100.50),  # bar 2 - gap-up open==low -> fill at 100.21
        (100.50, 100.70, 100.40, 100.60),  # bar 3 - tail
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
        f"exit should land on bar 2 (the gap-up open bar), got {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 100.21) < 1e-9, (
        f"trail should fill at the gap-up open 100.21, got {t.exit_price}"
    )
