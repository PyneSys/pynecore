"""
@pyne

Regression test for ``strategy.risk.max_intraday_filled_orders``, covering two
intraday-cap bugs:

* a position-reversing order is a SINGLE filled order — its close half and open
  half must increment the daily counter once, not twice, and
* when the cap is reached while a position is open, TradingView flattens it
  ("Close Position (Max number of filled orders in one day)"), and this must
  fire when the cap is hit by a reversal, not only by a plain entry.

With ``max_intraday_filled_orders(4)`` and a reversal every bar, four entries
fill within the single trading day; the fourth reverses straight into the cap
and is closed on the same bar, leaving four closed trades whose final exit is
the cap close. Before the fixes the reversal double-count admitted only ~two
entries before the cap, and the missing flip-path cap-close left the last entry
open instead of flat.
"""
from pynecore.lib import script, strategy, bar_index


# noinspection PyTypeChecker
@script.strategy(
    "Intraday Cap Reversal",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    strategy.risk.max_intraday_filled_orders(4)

    # Flip direction every bar — each fill after the first is a reversal.
    if bar_index % 2 == 0 and strategy.position_size <= 0:
        strategy.entry('L', strategy.long)
    if bar_index % 2 == 1 and strategy.position_size >= 0:
        strategy.entry('S', strategy.short)


def _make_syminfo(period: str = '1'):
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period=period, type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_reversal_counts_once_and_caps_close__(script_path, module_key):
    """
    ``max_intraday_filled_orders(4)`` with a reversal every bar admits four
    entry fills in the day; the fourth reverses into the cap and is closed on
    the same bar. Result: four closed trades, the last one closed by the cap.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC — every bar stays in this day

    # Ten one-minute bars, all on the same UTC day so the counter never resets.
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0 + i, high=101.0 + i,
              low=99.0 + i, close=100.5 + i, volume=100.0)
        for i in range(10)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 4, (
        f"Expected exactly 4 closed trades (4 entry fills under the cap of 4 — "
        f"the fourth reversal hits the cap and is flattened on the same bar), "
        f"got {len(trades)}. A reversal counted as two fills would have hit the "
        "cap after ~2 entries; a missing flip-path cap-close would have left the "
        "fourth entry open (3 closed)."
    )
    # Every entry was a single lot — the reversal slice did not decay.
    assert all(abs(t.size) == 1.0 for t in trades), (
        f"sizes={[t.size for t in trades]}"
    )
    # The fourth trade is closed by the intraday-cap flatten, like TradingView.
    assert trades[-1].exit_comment == "Close Position (Max number of filled orders in one day)", (
        f"last exit comment: {trades[-1].exit_comment!r}"
    )
    # The cap close is a same-bar wash (entry price == exit price).
    assert trades[-1].entry_price == trades[-1].exit_price, (
        f"cap close should be a same-bar wash: entry={trades[-1].entry_price} "
        f"exit={trades[-1].exit_price}"
    )
