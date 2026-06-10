"""
@pyne

Regression test for the stricter-activation re-arm with the ``trail_points`` form.

The re-arm exception (a re-issued trailing leg whose activation level moved above
the carried high-water mark re-arms instead of inheriting the armed state — see
test_060, observed against the TradingView reference on BINANCE:BTCUSDT) must
apply regardless of HOW the activation level is expressed. ``trail_points`` is
the entry-anchored tick form of the same activation level: TradingView resolves
it to ``entry ± points * mintick`` and runs the same one-logical-trailing-stop
model, so the stricter-level decision has to compare the RESOLVED price, not
just an explicit ``trail_price`` argument.

This mirrors test_060 exactly (entry fills at 100.00, mintick 0.01): 100 ticks
== activation 101.00 and 250 ticks == activation 102.50, so the expected trade
is identical. The pre-fix code skipped the stricter-level check whenever
``trail_price`` was not given explicitly and inherited the stale armed stop.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Re-Arm On Stricter Level (trail_points)",
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
        trail_points = 100 if bar_index < 2 else 250
        strategy.exit('X', 'L', trail_points=trail_points, trail_offset=10)


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
def __test_stricter_reissued_points_rearm_the_trail__(script_path, module_key):
    """
    Raising trail_points above the water mark must disarm the carried trail.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: exit issued with trail_points=100 (activation 101.00), offset=10
      ticks (0.10).
    * bar 2: arms at 101.00, water mark 101.50 (stop 101.40), close 101.45 —
      no fill. The bar-2 re-issue RAISES trail_points to 250 (activation
      102.50), above the 101.50 water mark: the trail must re-arm there.
    * bar 3: dips to 101.20 — through the stale 101.40 stop. No fill may
      happen (102.50 was never reached).
    * bar 4: runs to 102.80 (arming at 102.50), closes 102.55: the trail fills
      at 102.80 - 0.10 = 102.70.

    The pre-fix behavior inherited the armed state across the re-issue
    (the stricter-level check only looked at an explicit ``trail_price``)
    and filled on bar 3 at 101.40.
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
        (100.50, 101.50, 100.45, 101.45),  # bar 2 - arm @101, wm=101.50, re-issue @250 ticks
        (101.40, 101.80, 101.20, 101.60),  # bar 3 - dips through stale stop: no fill
        (101.60, 102.80, 101.50, 102.55),  # bar 4 - arm @102.50, ride, fill 102.70
        (102.55, 102.60, 102.40, 102.50),  # bar 5 - tail
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
    assert t.exit_bar_index == 4, (
        f"the trail must re-arm at 102.50 and fill on bar 4, got bar "
        f"{t.exit_bar_index} (bar 3 means the stale armed state was inherited)"
    )
    assert abs(t.exit_price - 102.70) < 1e-9, (
        f"fill should land at 102.80 - 0.10 = 102.70, got {t.exit_price}"
    )
