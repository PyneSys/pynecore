"""
@pyne

Regression test: a ``strategy.exit`` re-issued with a MOVED level replaces the leg.

Only an identical re-issue may be taken as a no-op. As soon as the script states a
different stop the resting order has to be replaced, so the old level stops
filling from the very next bar.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Exit Changed Level Replaces",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if 1 <= bar_index <= 2:
        strategy.exit('X', from_entry='L', qty=1, stop=95.0)
    if bar_index >= 3:
        strategy.exit('X', from_entry='L', qty=1, stop=90.0)


def __test_helper_syminfo():
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


def __test_helper_run(script_path, module_key, rows):
    """Run ``main`` over ``rows`` and return the closed trades."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)
    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=lo, close=c, volume=100.0)
        for i, (o, h, lo, c) in enumerate(rows)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
    return trades


# noinspection PyShadowingNames
def __test_exit_changed_level_replaces__(script_path, module_key):
    """
    Moving the stop replaces the resting leg: the old level stops filling at once.

    * bars 1-2: ``stop=95``.
    * bar 3 on: ``stop=90`` -- a changed level, so the early-out must not take it
      and the resting order is replaced.
    * bar 4 dips to 92: below the OLD 95 but above the new 90, so nothing fills.
    * bar 5 dips to 88 and the leg fills at the new level, 90.
    """
    rows = [
        # open,  high,  low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - 'L' fills @100, stop 95 placed
        (100.0, 100.5, 99.5, 100.0),  # bar 2 - stop 95 re-issued unchanged
        (100.0, 100.5, 99.5, 100.0),  # bar 3 - stop moved to 90
        (99.0, 99.5, 92.0, 93.0),     # bar 4 - below the old 95, above the new 90
        (93.0, 93.5, 88.0, 89.0),     # bar 5 - through the new 90
    ]
    trades = __test_helper_run(script_path, module_key, rows)

    assert len(trades) == 1, f"expected 1 closed trade, got {len(trades)}"
    t = trades[0]
    assert t.exit_bar_index == 5, (
        f"the moved stop must replace the resting leg, so bar 4 (low 92) cannot "
        f"fill the old 95 level; got exit bar {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 90.0) < 1e-9, f"exit price {t.exit_price}"
