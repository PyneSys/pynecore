"""
@pyne

An entry re-issued under an id that already has THIS BAR's unfilled order
MODIFIES that order — the pyramiding limit has nothing to judge.

The modification carries the raw quantity: the reversal flip the replaced order
was built with is not computed again (see ``skip_flip``). Rejecting the second
call as a pyramid add would leave the FIRST order standing, flip and all, and
reverse the whole position instead of the difference.

MEASURED on the wild `Strategy for UT Bot Alerts indicator` reference
(BINANCE:BTCUSDT 30m), whose two near-identical blocks issue the same entry twice
on every signal bar: TV reverses a 28.20314 long into a 2.10875 short — the
second call's flip-free quantity — not the 30.31189 the first call's flip opens.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Reissued Entry Modifies Standing Order",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    pyramiding=1,
)
def main():
    if bar_index == 0:
        strategy.entry('X', strategy.long)
    if bar_index == 2:
        # Both calls land on the same bar, so the second rewrites the first.
        strategy.entry('Y', strategy.short, 3)
        strategy.entry('Y', strategy.short, 3)
    if bar_index == 4:
        strategy.close_all()


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


def _run(script_path, module_key):
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000,
              open=100.0, high=100.05, low=99.95, close=100.0, volume=100.0)
        for i in range(8)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
    return trades


# noinspection PyShadowingNames
def __test_reissued_entry_reverses_by_the_raw_quantity__(script_path, module_key):
    """
    The 3-unit short lands as a 3-unit DELTA, leaving 2 short over the 1 long.

    Dropping the second call as a pyramid add would keep the first order, whose
    flip closes the long AND opens the full 3.
    """
    trades = _run(script_path, module_key)
    shape = [(t.entry_id, t.entry_bar_index, t.exit_bar_index, t.size) for t in trades]

    assert len(trades) == 2, f"expected two closed trades, got {shape}"
    long_trade, short_trade = trades

    assert (long_trade.entry_id, long_trade.entry_bar_index) == ('X', 1), shape
    assert abs(long_trade.size - 1.0) < 1e-9, shape

    assert (short_trade.entry_id, short_trade.entry_bar_index) == ('Y', 3), shape
    assert short_trade.sign < 0, shape
    assert abs(abs(short_trade.size) - 2.0) < 1e-9, shape
