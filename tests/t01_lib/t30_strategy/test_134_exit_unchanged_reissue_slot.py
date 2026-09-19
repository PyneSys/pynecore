"""
@pyne

Regression test: a ``strategy.exit`` leg re-issued UNCHANGED keeps its slot.

Re-stating a resting leg with the very same levels, quantity, comments and OCA
settings must leave that order in place -- including the activation slot
(``Order.act_seq``) that decides which of several orders triggering at ONE price
level fills first. An implementation that rebuilds, or re-indexes, the leg on
every bar would push it behind an order placed after it.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Exit Unchanged Re-issue Slot",
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
        # Re-issued UNCHANGED on every bar: the resting leg keeps the activation
        # slot it took on bar 1, ahead of the stop entry placed on bar 3.
        strategy.exit('X', from_entry='L', qty=1, stop=95.0)
    if bar_index == 3:
        strategy.entry('S', strategy.short, stop=95.0)


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
def __test_exit_unchanged_reissue_keeps_slot__(script_path, module_key):
    """
    A leg re-issued unchanged keeps its activation slot ahead of a later stop entry.

    * bar 0: entry signal -> 'L' fills at bar 1's open, 2 units at 100.
    * bars 1..: ``strategy.exit('X', qty=1, stop=95)`` re-issued unchanged every
      bar. The early-out leaves the resting order in place, so it must keep the
      ``act_seq`` it took on bar 1.
    * bar 3: a short stop entry is placed on the SAME level (95).
    * bar 4: opens at 98 and trades down through 95, so both orders trigger at
      that one level. They are ordered by ``act_seq``, so the exit -- issued two
      bars earlier -- fills first and its trade is the first closed one.
    """
    rows = [
        # open,  high,  low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - 'L' fills @100, 'X' placed
        (100.0, 100.5, 99.5, 100.0),  # bar 2 - 'X' re-issued unchanged
        (100.0, 100.5, 99.5, 100.0),  # bar 3 - 'X' re-issued, short stop @95 placed
        (98.0, 98.5, 88.0, 92.0),     # bar 4 - walks down through 95
        (92.0, 92.5, 91.5, 92.0),     # bar 5 - tail
    ]
    trades = __test_helper_run(script_path, module_key, rows)

    assert trades, "expected at least one closed trade"
    first = trades[0]
    assert first.exit_id == 'X', (
        f"the unchanged re-issued exit must keep its slot and fill first, "
        f"got {first.exit_id!r} first"
    )
    assert first.exit_bar_index == 4, f"X exit bar {first.exit_bar_index}"
    assert abs(first.exit_price - 95.0) < 1e-9, f"X exit price {first.exit_price}"
