"""
@pyne

A reversal rejected at its fill withdraws the standing exits of the position it leaves open.

A reversing market ``strategy.entry`` that could be margined at the placement close
but not at the fill price is rejected whole and the old position stays. MEASURED on
TradingView (BINANCE:BTCUSDT 30m, a long with a stop/limit ``strategy.exit`` bracket
and a short entry three bars later): in 4 of 497 cycles the short is rejected at the
next open, and the long then rides through its bracket levels -- intrabar and across
an opening gap -- until the cycle's ``strategy.close_all``.
"""
from pynecore.lib import script, strategy, bar_index


@script.strategy(
    "Rejected Reversal Withdraws Exits",
    overlay=True,
    initial_capital=1000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
)
def main():
    if bar_index == 0:
        strategy.entry('L', strategy.long, qty=1)
        strategy.exit('XL', 'L', stop=95.0, limit=105.0)
    if bar_index == 2:
        strategy.entry('S', strategy.short, qty=9.95)
    if bar_index == 7:
        strategy.close_all()
    return {"psize": strategy.position_size}


def __test_helper_make_syminfo():
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
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [
        OHLCV(timestamp=base_ts + i * 60_000, open=o, high=h, low=l, close=c, volume=100.0)
        for i, (o, h, l, c) in enumerate(rows)
    ]
    runner = ScriptRunner(Path(script_path), iter(bars), __test_helper_make_syminfo())
    trades = []
    sizes = []
    for _candle, plot, new_closed in runner.run_iter():
        trades.extend(new_closed)
        sizes.append(round(plot["psize"], 6))
    return trades, sizes


# noinspection PyShadowingNames
def __test_rejected_reversal_withdraws_exits__(script_path, module_key):
    """
    The long outlives its bracket once the short is rejected at the gapped open.

    * bar 2: the 9.95-contract short needs 995 at the close (100.00) -- affordable.
    * bar 3: the open (101.00) lifts the need to 1004.95 against 1001 of equity; the
      short is rejected and the long keeps running.
    * bar 4: the high passes the 105.00 limit -- no fill.
    * bar 5: the low passes the 95.00 stop -- no fill.
    * bar 6: the open gaps below the stop -- no fill.
    * bar 8: ``close_all`` closes the long at the open (94.50).
    """
    rows = [
        # open,   high,   low,    close
        (100.00, 100.05, 99.95, 100.00),  # bar 0 - long and bracket placed
        (100.00, 100.05, 99.95, 100.00),  # bar 1 - long fills
        (100.00, 100.05, 99.95, 100.00),  # bar 2 - short placed
        (101.00, 101.05, 100.95, 101.00),  # bar 3 - short rejected at the open
        (101.00, 106.00, 100.95, 101.00),  # bar 4 - limit passed
        (101.00, 101.05, 94.00, 96.00),  # bar 5 - stop passed
        (94.50, 94.55, 94.45, 94.50),  # bar 6 - open below the stop
        (94.50, 94.55, 94.45, 94.50),  # bar 7 - close_all placed
        (94.50, 94.55, 94.45, 94.50),  # bar 8 - long closed
        (94.50, 94.55, 94.45, 94.50),  # bar 9 - flat
    ]
    trades, sizes = __test_helper_run(script_path, module_key, rows)

    assert sizes == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], sizes
    shape = [(t.entry_id, t.entry_bar_index, t.exit_bar_index) for t in trades]
    assert shape == [('L', 1, 8)], shape
    assert abs(trades[0].exit_price - 94.50) < 1e-9, trades[0].exit_price
