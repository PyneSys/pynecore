"""
@pyne

Regression test for a default-sized entry's bracket exit covering the WHOLE fill.

A default-sized (percent_of_equity / cash) price-based entry sizes its quantity
at PLACEMENT only as a margin-check estimate (a buy stop uses ``max(stop,
close)``), then re-resolves to the real quantity at the actual fill price. Its
no-qty bracket exit (``strategy.exit(id, from_entry)`` — the "rest" leg that
closes the whole entry) reserved off that placement estimate. When the fill
price is LOWER than the estimate price (the stop fills below a higher placement
close), the entry re-sizes UP while the exit stayed frozen, so the exit
under-closed the fill and stranded a sliver that surfaced as a spurious second
trade. Observed on the wild corpus (Divergence Strategy [Trendoscope®],
BINANCE:BTCUSDT 30m: TV one entry of 2.59141, PyneCore 2.59141 + a 0.00528
sliver closed later by ``strategy.close``).
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Deferred Entry Exit Coverage",
    overlay=True,
    initial_capital=1000000,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=20,
    commission_type=strategy.commission.percent,
    commission_value=0.1,
    pyramiding=4,
)
def main():
    # Buy stop BELOW the bar-0 close (close 110, stop 105): the placement
    # estimate sizes at 110, the fill next bar lands at 105 -> larger deferred
    # quantity. The no-qty bracket exit must cover that full fill.
    if bar_index == 0:
        strategy.entry("L", strategy.long, stop=105.0)
        strategy.exit("X", "L", stop=90.0, limit=200.0)

    # Force any stranded sliver to surface as a closed trade.
    if bar_index == 4:
        strategy.close("L")


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
def __test_deferred_entry_closed_in_one_trade__(script_path, module_key):
    """
    A default-sized entry that re-sizes UP at fill is closed in a SINGLE trade.

    * bar 0: buy stop at 105 (below the 110 close) + a no-qty bracket exit.
      The placement estimate sizes the entry at 110 (qty ~1816.37).
    * bar 1: opens at 105 -> the stop fills at 105, re-sizing UP to ~1902.86.
    * bar 2: high 205 crosses the 200 limit -> the whole position exits at 200.
    * bar 4: strategy.close is a no-op if nothing was stranded.

    The pre-fix behavior closed only the frozen estimate slice (1816.37) at 200
    and left the ~86.5 sliver open, which strategy.close then closed at 205 as a
    spurious second trade.
    """
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    rows = [
        # open,   high,   low,    close
        (110.00, 110.00, 110.00, 110.00),  # bar 0 - entry+exit signal; estimate at 110
        (105.00, 160.00, 105.00, 160.00),  # bar 1 - buy stop fills at 105 (re-sizes up)
        (160.00, 205.00, 160.00, 205.00),  # bar 2 - limit 200 hit -> full exit at 200
        (205.00, 205.00, 205.00, 205.00),  # bar 3
        (205.00, 205.00, 205.00, 205.00),  # bar 4 - strategy.close mop-up (no-op when clean)
        (205.00, 205.00, 205.00, 205.00),  # bar 5 - tail
    ]
    bars = [
        OHLCV(timestamp=base_ts + i * 60, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    trades = []
    for _candle, _plot, new_closed in runner.run_iter():
        trades.extend(new_closed)

    assert len(trades) == 1, (
        f"deferred entry must close in one trade, got {len(trades)} "
        f"(stranded sliver): {trades}"
    )

    trade = trades[0]
    assert abs(trade.entry_price - 105.00) < 1e-9, f"entry_price={trade.entry_price}"
    assert abs(trade.exit_price - 200.00) < 1e-9, f"exit_price={trade.exit_price}"

    # The single trade must carry the FULL fill-time quantity, not the smaller
    # placement estimate (200000 / (110 * 1.001) ~= 1816.37).
    estimate_qty = 200000.0 / (110.00 * 1.001)
    full_fill_qty = 200000.0 / (105.00 * 1.001)
    assert abs(trade.size) > estimate_qty, (
        f"exit closed only the frozen estimate slice: size={trade.size}"
    )
    assert abs(abs(trade.size) - full_fill_qty) < 1e-3, (
        f"trade must close the whole deferred fill, got {trade.size} "
        f"(expected ~{full_fill_qty})"
    )
