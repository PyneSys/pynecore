"""
@pyne

Regression test for the carried short trailing-stop fill on a gap-down open bar.

Mirror of the long case: a short position's trailing stop (``trail_offset=0``)
rides the low-water mark. When the NEXT bar opens at its own high (no upper wick)
and that open gaps below the carried water mark, TradingView folds the open into
the low-water mark and the trail fills at the open -- the bar is already at its
highest there and never trades back up. This branch is the symmetric counterpart
of the long fill and is exercised here deterministically (the live PineForge
ETHUSDT probes only happen to surface the long side).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Gap-Open Short",
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
        strategy.exit('X', 'S', trail_points=5, trail_offset=0)


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
def __test_gap_open_short_fills_at_open__(script_path, module_key):
    """
    The carried short trail fills at the gap-down open of a no-upper-wick bar.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: drops to a low of 99.80 and closes there, arming the trail and
      ratcheting the offset-0 water mark to 99.80 (no same-bar fill: the close
      is the low, so the bar never retraces above it).
    * bar 2: opens at its own high 99.79 -- one tick below the carried water
      mark -- then drops further. The trail must fill at the open, 99.79.

    Without the open-water-mark fold the stop stays at 99.80, the bar's high
    (99.79) never reaches it, and the trail rides down to 99.40 instead.
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
        (100.00, 100.00, 99.80, 99.80),    # bar 1 - entry fill, trail arms, wm=99.80
        (99.79, 99.79, 99.40, 99.50),      # bar 2 - gap-down open==high -> fill at 99.79
        (99.50, 99.60, 99.30, 99.40),      # bar 3 - tail
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
        f"exit should land on bar 2 (the gap-down open bar), got {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 99.79) < 1e-9, (
        f"trail should fill at the gap-down open 99.79, got {t.exit_price}"
    )
