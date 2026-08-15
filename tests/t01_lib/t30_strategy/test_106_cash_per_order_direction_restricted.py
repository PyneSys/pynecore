"""
@pyne

Regression test: when ``strategy.risk.allow_entry_in()`` suppresses the opening leg of a
reversal, the order executes as a plain close -- and that close pays the whole
``cash_per_order`` fee, not the share it would have taken in a real reversal.

Measured on TradingView with CommCashOrderDirProbe (BINANCE:BTCUSDT 30m, fee 10, one
contract, allow_entry_in(long), long -> short -> long -> close_all): the short entry
only closes the long, both closed trades report a commission of 20 (a whole fee on the
entry plus a whole fee on the close), and the first trade's profit is its gross 639.24
minus exactly 20.
"""
from pynecore.lib import script, strategy, plot, bar_index


@script.strategy(
    "Cash Per Order Direction Restricted",
    initial_capital=1_000_000,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    commission_type=strategy.commission.cash_per_order,
    commission_value=10,
    margin_long=0,
    margin_short=0,
    process_orders_on_close=True,
)
def main():
    strategy.risk.allow_entry_in(strategy.direction.long)

    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if bar_index == 1:
        strategy.entry('S', strategy.short)
    if bar_index == 2:
        strategy.entry('L2', strategy.long)
    if bar_index == 3:
        strategy.close_all()

    plot(strategy.netprofit, "netprofit")
    plot(strategy.closedtrades, "closedtrades")
    plot(strategy.closedtrades.commission(0), "comm0")
    plot(strategy.closedtrades.commission(1), "comm1")
    plot(strategy.closedtrades.profit(0), "profit0")
    plot(strategy.closedtrades.profit(1), "profit1")


def _make_syminfo():
    from pynecore.core.syminfo import SymInfo
    from pynecore.providers.ccxt import CCXTProvider
    # noinspection PyProtectedMember
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()
    return SymInfo(
        prefix="TEST", description="Test", ticker="TEST", currency="USD",
        period='30', type="crypto", mintick=0.01, pricescale=100,
        minmove=1, pointvalue=1, timezone="UTC", volumetype="base",
        mincontract=0.0001,
        opening_hours=opening_hours, session_starts=session_starts,
        session_ends=session_ends,
    )


# noinspection PyShadowingNames
def __test_suppressed_reversal_pays_the_whole_flat_fee__(script_path, module_key):
    """A close-only transaction is still one order, so it owes one whole fee."""
    import sys
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(module_key, None)

    base_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    closes = [100.0, 110.0, 105.0, 120.0, 120.0]
    bars = [
        OHLCV(timestamp=base_ts + i * 1_800_000, open=c, high=c, low=c, close=c, volume=1.0)
        for i, c in enumerate(closes)
    ]

    runner = ScriptRunner(Path(script_path), iter(bars), _make_syminfo())
    plots = [dict(p) for _candle, p, _closed in runner.run_iter()]

    last = plots[-1]
    # The short entry cannot open, so only two trades exist.
    assert last["closedtrades"] == 2.0, last["closedtrades"]

    # Whole fee on the entry plus a whole fee on the close, on both trades.
    assert last["comm0"] == 20.0, last["comm0"]
    assert last["comm1"] == 20.0, last["comm1"]

    # Gross 10 and 15, minus 20 of fees each.
    assert last["profit0"] == -10.0, last["profit0"]
    assert last["profit1"] == -5.0, last["profit1"]

    # Four orders were placed, but only three transacted -- 30 in fees, 25 gross.
    assert last["netprofit"] == -15.0, last["netprofit"]
