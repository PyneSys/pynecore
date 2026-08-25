"""
@pyne

Regression test: a trailing exit first issued AFTER its entry filled anchors its
water mark to the ISSUE BAR'S CLOSE, not to that bar's extreme.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, a probe entering long on every 97th
bar and issuing ``strategy.exit(trail_points=50, trail_offset=10)`` from inside
``if strategy.position_size > 0``, so the leg is first issued on the entry-FILL
bar): all 222 exits land on the close-anchored price. Folding the entry-fill bar's
high into the mark instead puts the stop above the next bar's open, which then gaps
through and fills at that open -- 106 of the 222 exits are wrong that way. The
control probe, whose exit is issued unconditionally (so it is already live during
the entry-fill bar's own walk), is unaffected and matches either way.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Trailing First Issue Anchors Close",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('L1', strategy.long)
    if strategy.position_size > 0:
        strategy.exit('L1X', from_entry='L1', trail_points=10, trail_offset=10)


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
def __test_issue_bar_high_stays_out_of_the_water_mark__(script_path, module_key):
    """
    The issue bar's high must not lift the stop above the next bar's open.

    * bar 0: the market entry is placed.
    * bar 1: it fills at the open 100.00; the script then sees the position and
      issues the trailing exit. Activation sits 10 ticks above the entry at
      100.10, and the bar closes at 100.30 -- past it -- so the leg arms here.
      The mark is the CLOSE (100.30), giving a stop of 100.20; the bar's own high
      of 101.00 belongs to a stretch the leg was not live for.
    * bar 2: the drop from 100.25 to 100.00 crosses that stop and fills it at
      100.20.

    Folding bar 1's high in would put the stop at 100.90, above bar 2's open, so
    the exit would gap-fill at that open (100.25) instead -- the two models are
    told apart by the exit price alone.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    rows = [
        # open,   high,   low,    close
        (100.00, 100.05, 99.95, 100.00),  # bar 0 - market entry placed
        (100.00, 101.00, 99.90, 100.30),  # bar 1 - entry fills at the open, exit issued
        (100.25, 100.25, 100.00, 100.10),  # bar 2 - stop follows the open, fills at 100.15
        (100.10, 100.20, 100.00, 100.15),  # bar 3 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
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
    assert t.exit_bar_index == 2, f"exit_bar_index={t.exit_bar_index}"
    assert abs(t.exit_price - 100.20) < 1e-9, (
        f"the mark anchors to bar 1's close (100.30), so the stop sits 10 ticks "
        f"under it at 100.20; 100.25 would mean bar 1's high armed the leg and the "
        f"exit gapped through bar 2's open, got {t.exit_price}"
    )
