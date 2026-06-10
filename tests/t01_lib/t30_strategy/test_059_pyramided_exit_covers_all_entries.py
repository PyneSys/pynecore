"""
@pyne

Regression test for one strategy.exit covering every pyramided same-ID entry.

A ``strategy.exit(from_entry='L')`` bracket applies to EVERY trade entered with
that ID — TradingView exits all pyramid adds together (one trade-list row per
entry, same fill). PyneCore previously kept a single exit leg keyed
(id, from_entry) whose reservation a pending pyramid add overwrote with the
NEW entry's size, so the exit under-closed the position and stranded the older
trade (plus a rounding sliver) for some other rule to mop up later. Observed
against the TradingView reference on BINANCE:BTCUSDT (2025-01-17 14:30: TV
closed 0.00392 + 0.00391 together; PyneCore closed only 0.0039 and left
0.00002 + 0.0039 open).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Pyramided Exit Coverage",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=2,
)
def main():
    if bar_index == 0 and strategy.position_size == 0:
        strategy.entry('L', strategy.long)
    if bar_index == 2:
        strategy.entry('L', strategy.long)
    if strategy.position_size > 0:
        strategy.exit('X', 'L', stop=99.00)


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
def __test_exit_closes_both_pyramided_entries__(script_path, module_key):
    """
    The stop bracket bound to 'L' must close BOTH pyramid adds together.

    * bar 0: first entry signal -> fills bar 1 open at 100.00 (qty 1).
    * bar 2: second entry signal (pyramiding=2) -> fills bar 3 open at 101.00
      (qty 1). The re-issued exit's reservation must now cover both entries.
    * bar 4: drops through the 99.00 stop -> both trades exit at 99.00 on the
      same bar.

    The pre-fix behavior reserved only the second entry's size (1), closing
    just the first trade and leaving the second one stranded.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    rows = [
        # open,   high,   low,    close
        (100.00, 100.00, 100.00, 100.00),  # bar 0 - entry signal #1
        (100.00, 100.50, 100.00, 100.40),  # bar 1 - entry #1 fills at 100.00
        (100.40, 101.00, 100.30, 101.00),  # bar 2 - entry signal #2
        (101.00, 101.20, 100.80, 101.00),  # bar 3 - entry #2 fills at 101.00
        (101.00, 101.10, 98.50, 98.80),    # bar 4 - stop 99.00 hit
        (98.80, 99.00, 98.50, 98.70),      # bar 5 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, (
        f"expected both pyramided trades to close, got {len(trades)}: {trades}"
    )
    first, second = sorted(trades, key=lambda t: t.entry_bar_index)
    assert first.entry_bar_index == 1 and abs(first.entry_price - 100.00) < 1e-9
    assert second.entry_bar_index == 3 and abs(second.entry_price - 101.00) < 1e-9
    for t in (first, second):
        assert t.exit_bar_index == 4, (
            f"both trades must exit on bar 4, got {t.exit_bar_index} for "
            f"entry bar {t.entry_bar_index}"
        )
        assert abs(t.exit_price - 99.00) < 1e-9, f"exit_price={t.exit_price}"
        assert abs(abs(t.size) - 1.0) < 1e-9, (
            f"each trade must close at its full size, got {t.size}"
        )
