"""
@pyne

Regression test for the per-bar partial-exit re-arm (resurrection) bug.

A strategy that calls ``strategy.exit()`` on *every* bar while the position is
open (TradingView's standard sticky-bracket pattern) must behave like TV:

* a ``qty_percent`` leg reserves a fixed slice of the entry's ORIGINAL size and
  fires that slice exactly once — re-calling it after it filled must NOT
  resurrect it with a freshly-recomputed (shrinking) size, and
* a no-``qty`` "rest" leg closes the entry size MINUS the slices reserved by its
  sibling legs — i.e. one lot here, not the whole remaining position, even when
  it fires before the ``qty_percent`` leg.

Before the fix, the per-bar replacement re-armed a filled ``qty_percent`` leg at
50% of the shrinking remainder (2 -> 1 -> 0.5 -> 0.25 ...), exploding the trade
count, and the no-qty leg always closed the full remainder.
"""
from pynecore.lib import script, strategy


@script.strategy(
    "Sticky Partial Bracket",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=1,
)
def main():
    if strategy.position_size == 0:
        strategy.entry('L', strategy.long)

    # Re-issued on every bar while in position — the sticky-bracket pattern.
    if strategy.position_size > 0:
        entry = strategy.position_avg_price
        strategy.exit('HALF_TP', from_entry='L', limit=entry * 1.05, qty_percent=50)
        strategy.exit('REST_SL', from_entry='L', stop=entry * 0.95)


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    opening_hours, session_starts, session_ends = CCXTProvider.get_opening_hours_and_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_qty_percent_leg_not_resurrected__(script_path, module_key):
    """
    The HALF_TP (qty_percent=50) leg fires once for 1 lot; staying above the
    limit on later bars must not re-fire it at 0.5, 0.25, ... lots.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    # Entry fills at bar 1 open=100 (avg=100 -> limit 105, stop 95).
    # Bar 2 high 106 -> HALF_TP fills 1 lot. Bars 3-5 stay above 105 and well
    # above the stop, so a correct sticky bracket fires nothing more.
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=101.0, high=106.0, low=100.5, close=105.0, volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=106.0, high=107.0, low=105.5, close=106.0, volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60, open=106.0, high=108.0, low=105.5, close=107.0, volume=100.0),
        OHLCV(timestamp=base_ts + 5 * 60, open=106.0, high=108.0, low=105.5, close=107.0, volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, (
        f"Expected exactly 1 closed trade (HALF_TP, 1 lot), got {len(trades)} "
        f"sizes={[t.size for t in trades]} — the filled qty_percent leg was "
        "resurrected by the per-bar re-call."
    )
    assert trades[0].exit_id == 'HALF_TP', f"exit id: {trades[0].exit_id}"
    assert trades[0].size == 1.0, f"HALF_TP size: {trades[0].size} (expected 1.0)"
    assert trades[0].exit_price == 105.0, f"HALF_TP exit price: {trades[0].exit_price}"


# noinspection PyShadowingNames
def __test_no_qty_rest_leg_closes_only_its_slice__(script_path, module_key):
    """
    When the no-qty REST_SL leg fires first, it closes only its reserved slice
    (entry 2 - HALF_TP's reserved 1 = 1 lot), not the whole 2-lot position.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200

    # Entry fills at bar 1 open=100 (stop 95, limit 105). Bar 2 low 94 hits the
    # stop first; REST_SL must close 1 lot, leaving HALF_TP's 1 lot open.
    bars = [
        OHLCV(timestamp=base_ts + 0 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 1 * 60, open=100.0, high=100.5, low=99.5,  close=100.0, volume=100.0),
        OHLCV(timestamp=base_ts + 2 * 60, open=99.0,  high=99.5,  low=94.0,  close=95.0,  volume=100.0),
        OHLCV(timestamp=base_ts + 3 * 60, open=95.0,  high=96.0,  low=94.5,  close=95.5,  volume=100.0),
        OHLCV(timestamp=base_ts + 4 * 60, open=95.0,  high=96.0,  low=94.5,  close=95.5,  volume=100.0),
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, (
        f"Expected exactly 1 closed trade (REST_SL, 1 lot), got {len(trades)} "
        f"sizes={[t.size for t in trades]}"
    )
    assert trades[0].exit_id == 'REST_SL', f"exit id: {trades[0].exit_id}"
    assert trades[0].size == 1.0, (
        f"REST_SL size: {trades[0].size} (expected 1.0 — the no-qty rest leg "
        "must close entry-minus-sibling, not the full position)"
    )
