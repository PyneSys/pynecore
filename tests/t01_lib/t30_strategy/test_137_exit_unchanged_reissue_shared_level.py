"""
@pyne

Regression test: an unchanged ``strategy.exit`` re-issue still queues behind a
later order resting on the same price level.

Every re-issue of a leg re-indexes it in the price book, which puts it at the
BACK of the orders sharing its level. While the leg is alone on its level that
changes nothing, but once another order rests there the leg drops behind it,
and a bar that GAPS through the level fills the orders in exactly that book
order. A leg kept in place without that re-queue would fill ahead of the order
it has to follow.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Exit Unchanged Re-issue Shared Level",
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
        strategy.exit('A', from_entry='L', qty=1, stop=95.0)
    if bar_index == 3:
        strategy.exit('B', from_entry='L', qty=1, stop=95.0)


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
def __test_exit_unchanged_reissue_requeues_on_shared_level__(script_path, module_key):
    """
    On a gap bar the leg issued once fills ahead of the leg re-issued behind it.

    * bar 1: 'L' fills at 100, 2 units; 'A' (qty 1, stop 95) rests alone on its level.
    * bar 3: 'A' is re-issued, then 'B' (qty 1, stop 95) is issued ONCE and lands
      on the same level behind it.
    * bar 4: 'A' is re-issued unchanged while 'B' shares its level, which queues
      'A' behind 'B'.
    * bar 5: opens at 90, below the level. The gap fills the level in book order:
      'B' first, then 'A'.
    """
    rows = [
        # open,  high,  low,   close
        (100.0, 100.5, 99.5, 100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5, 100.0),  # bar 1 - 'L' fills @100, 'A' placed
        (100.0, 100.5, 99.5, 100.0),  # bar 2 - 'A' re-issued unchanged
        (100.0, 100.5, 99.5, 100.0),  # bar 3 - 'A' re-issued, 'B' placed on its level
        (100.0, 100.5, 99.5, 100.0),  # bar 4 - 'A' re-issued behind 'B'
        (90.0, 90.5, 89.5, 90.0),     # bar 5 - gaps down through 95
        (90.0, 90.5, 89.5, 90.0),     # bar 6 - tail
    ]
    trades = __test_helper_run(script_path, module_key, rows)

    assert [trade.exit_id for trade in trades] == ['B', 'A'], (
        f"the re-issued leg must queue behind the leg sharing its level, "
        f"got {[trade.exit_id for trade in trades]!r}"
    )
    for trade in trades:
        assert trade.exit_bar_index == 5, f"{trade.exit_id} exit bar {trade.exit_bar_index}"
        assert abs(trade.exit_price - 90.0) < 1e-9, f"{trade.exit_id} exit price {trade.exit_price}"
        assert abs(abs(trade.size) - 1.0) < 1e-9, f"{trade.exit_id} size {trade.size}"
