"""
@pyne

Regression test for the ``market_orders`` dict-key collision on a gap-through open.

Sibling of :mod:`test_040_partial_tp_collision` (which covers the ``exit_orders``
key collision). Two ``strategy.exit()`` legs share the same ``from_entry`` ("L").
When a bar's OPEN gaps past BOTH legs' limit levels at once, both convert to
market orders in ``_process_at_bar_open`` and fill at the open on that bar.

Before the fix, ``market_orders`` was keyed ``(order_type, order_id)`` — and both
legs share ``order_id`` (= ``from_entry`` = "L") — so the second gap-through leg
EVICTED the first from the dict. Only one leg filled at the gap open; the other
stayed pending and filled on a LATER bar at a different price. The key now
includes ``exit_id`` so the two legs are distinct.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy(
    "Gap-Open Multi-Bracket",
    overlay=True,
    initial_capital=100000,
    default_qty_type=strategy.fixed,
    default_qty_value=2,
    pyramiding=0,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    # Two take-profit legs on the same entry, placed once on the bar after the
    # fill so each is a standalone order rather than a per-bar replacement.
    if bar_index == 1:
        strategy.exit('X_A', from_entry='L', qty=1, limit=110.0)
        strategy.exit('X_B', from_entry='L', qty=1, limit=120.0)


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
def __test_gap_open_fills_both_brackets_same_bar__(script_path, module_key):
    """
    A gap-up open past both take-profit limits fills BOTH legs at the open, same bar.

    * bar 0: entry signal -> fills bar 1 open at 100.
    * bar 1: both legs placed (X_A limit 110, X_B limit 120).
    * bar 2: opens at 125 -- above BOTH limits -> both gap through and fill at the
      open 125 on bar 2.

    Before the fix the second gap-through leg evicted the first from
    ``market_orders``; only one filled on bar 2 and the other carried to bar 3
    (open 123) and filled there at 123.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    syminfo = _make_syminfo(period='1')
    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC

    rows = [
        # open,   high,   low,    close
        (100.0, 100.5, 99.5,  100.0),  # bar 0 - entry signal
        (100.0, 100.5, 99.5,  100.0),  # bar 1 - entry fill @100, both legs placed
        (125.0, 125.5, 124.5, 125.0),  # bar 2 - gap-up open past both limits -> both fill @125
        (123.0, 123.5, 122.5, 123.0),  # bar 3 - where a missed leg would fill (123)
        (123.0, 123.5, 122.5, 123.0),  # bar 4 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), syminfo)
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 2, (
        f"Expected 2 closed trades (both brackets fill on the gap bar), got "
        f"{len(trades)} — market_orders dict-key collision likely regressed."
    )
    by_id = {t.exit_id: t for t in trades}
    assert sorted(by_id) == ['X_A', 'X_B'], f"exit ids: {sorted(by_id)}"
    for eid in ('X_A', 'X_B'):
        assert by_id[eid].exit_bar_index == 2, (
            f"{eid} should fill on the gap bar 2, got bar {by_id[eid].exit_bar_index}"
        )
        assert abs(by_id[eid].exit_price - 125.0) < 1e-9, (
            f"{eid} should fill at the gap open 125.0, got {by_id[eid].exit_price}"
        )
