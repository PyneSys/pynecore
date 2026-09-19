"""
@pyne

Regression test: an unchanged re-issued trailing leg keeps its carried water mark.

TradingView carries ONE logical trailing stop across identical re-issues, so a
leg re-stated every bar with the same ``trail_points``/``trail_offset`` must go on
ratcheting its high-water mark instead of re-arming at the bare activation level.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Exit Unchanged Re-issue Trail",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if bar_index >= 1:
        strategy.exit('T', from_entry='L', trail_points=100.0, trail_offset=50.0)


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
def __test_exit_unchanged_reissue_keeps_trail_water_mark__(script_path, module_key):
    """
    An unchanged re-issued trailing leg keeps ratcheting its carried water mark.

    ``trail_points=100`` ticks and ``trail_offset=50`` ticks on a 0.01 mintick
    symbol arm the leg 1.00 above the 100.0 entry and trail 0.50 under the high.

    * bar 2 closes at 103 -> armed, stop 102.50.
    * bar 3 closes at 104 -> the carried mark ratchets to 104, stop 103.50.
    * bar 4 trades down to 102, so the leg fills at the RATCHETED 103.50. Had the
      unchanged re-issue dropped the carried mark and re-armed at the activation
      level, the stop would still sit at 100.50 and bar 4 would not fill at all.
    """
    rows = [
        # open,  high,  low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - 'L' fills @100, 'T' placed
        (100.0, 103.0, 100.0, 103.0),  # bar 2 - arms, stop 102.50
        (103.0, 104.0, 103.0, 104.0),  # bar 3 - ratchets, stop 103.50
        (104.0, 104.0, 102.0, 102.5),  # bar 4 - fills at 103.50
        (102.5, 103.0, 102.0, 102.5),  # bar 5 - tail
    ]
    trades = __test_helper_run(script_path, module_key, rows)

    assert len(trades) == 1, f"expected 1 closed trade, got {len(trades)}"
    t = trades[0]
    assert t.exit_bar_index == 4, f"trailing exit bar {t.exit_bar_index}"
    assert abs(t.exit_price - 103.5) < 1e-9, (
        f"the carried water mark must survive the unchanged re-issue: "
        f"expected an exit at 103.50, got {t.exit_price}"
    )
