"""
@pyne

Regression: several same-bar ``strategy.close()`` calls on one entry id (TP
ladder) must all fill. They share the synthetic ``exit_id`` + ``order_id``, so
before the fix they collided on the order-book key and the later call evicted the
earlier — only one slice closed. The fix stamps each backtest close with a unique
``book_seq`` so same-bar closes stack: 30% + 30% + 40% of 10 shed 3 + 3 + 4.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Same-Bar Close Ladder",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=10,
    pyramiding=0,
)
def main():
    # Entry on bar 0 (fills at bar 1 open).
    if bar_index == 0:
        strategy.entry('Long', strategy.long)
    # Three partial closes on ONE bar against the same entry id.
    if bar_index == 1:
        strategy.close('Long', 'TP1', qty_percent=30)
        strategy.close('Long', 'TP2', qty_percent=30)
        strategy.close('Long', 'TP3', qty_percent=40)


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
def __test_same_bar_qty_percent_ladder_stacks_and_closes_full__(script_path, module_key):
    """30% + 30% + 40% on one bar produce three closes (3 + 3 + 4) that flatten the position.

    On the buggy code path the three same-id closes collided on a shared exit key,
    so only the last (40% = 4) survived: a single closed trade, 6 units stranded.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=100.0, high=100.5, low=99.5,
              close=100.0, volume=100.0)
        for i in range(4)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    closed = [t for t in trades if t.size != 0.0]
    assert len(closed) == 3, (
        f"Expected 3 stacked ladder closes, got {len(closed)} — same-bar "
        "strategy.close() collision regressed (later call evicted earlier)."
    )
    sizes = sorted(abs(t.size) for t in closed)
    assert sizes == [3.0, 3.0, 4.0], f"Expected slices [3, 3, 4], got {sizes}"
    assert abs(sum(abs(t.size) for t in closed) - 10.0) < 1e-9, \
        "The ladder must close the whole 10-unit position."
