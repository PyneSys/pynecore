"""
@pyne

Regression test for a CARRIED long trailing stop ratcheting intrabar.

Once armed, the trailing stop follows the assumed intrabar path on every later
bar too: when a bar first runs to a new high (open -> high -> low -> close
path), the high-water mark advances to that new high BEFORE the falling leg is
evaluated, so the fill lands at ``new_high - offset`` — not at the stop level
carried in from the previous bar. PyneCore previously checked the bar only
against the carried stop, filling a full ratchet step too low.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing Carried Ratchet Long",
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
def __test_carried_trail_ratchets_with_current_bar_high__(script_path, module_key):
    """
    The carried trail must ratchet on the current bar's high before filling.

    * bar 0: entry signal (market) -> fills bar 1 open at 100.00.
    * bar 1: quiet bar, the exit is issued (trail_price=101.00, offset=10 ticks
      = 0.10).
    * bar 2: arms at 101.00 and ratchets the water mark to the high 101.20
      (stop 101.10); the close 101.15 stays above the stop, no fill.
    * bar 3: open 101.45 -> high 101.60 -> low 100.90 -> close 101.00. The open
      and then the high advance the water mark to 101.60 (stop 101.50) before
      the falling leg, which then fills at 101.50.

    The pre-fix behavior filled at the carried stop 101.10.
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
        (100.50, 101.20, 100.45, 101.15),  # bar 2 - arm, wm=101.20, no fill
        (101.45, 101.60, 100.90, 101.00),  # bar 3 - wm rides to 101.60, fill
        (101.00, 101.10, 100.80, 100.90),  # bar 4 - tail
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
    assert t.exit_bar_index == 3, (
        f"exit should land on bar 3, got {t.exit_bar_index}"
    )
    assert abs(t.exit_price - 101.50) < 1e-9, (
        f"trail should ratchet to the bar high and fill at 101.60 - 0.10 = 101.50, "
        f"got {t.exit_price}"
    )
