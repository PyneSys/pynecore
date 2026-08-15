"""
@pyne

Regression test: a ``cash_per_order`` fee is charged once per TradingView order, so a
reversal -- executed here as a closing and an opening fill -- pays it once and splits
it over the two legs in quantity proportion.

Measured on TradingView with CommCashOrderProbe (BINANCE:BTCUSDT 30m, fee 10, one
contract, long -> short -> long -> close_all): netprofit steps by exactly the gross
P&L minus 10 on every reversal bar, the run books 40 in total over its 4 orders, and
the three closed trades report commissions of 15, 10 and 15 -- each reversal splits
its single fee 5/5 between the leg it closed and the leg it opened.
"""
from pynecore.lib import script, strategy, plot, bar_index


@script.strategy(
    "Cash Per Order Reversal",
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
    if bar_index == 0:
        strategy.entry('L', strategy.long)
    if bar_index == 1:
        strategy.entry('S', strategy.short)
    if bar_index == 2:
        strategy.entry('L2', strategy.long)
    if bar_index == 3:
        strategy.close_all()

    plot(strategy.netprofit, "netprofit")
    plot(strategy.closedtrades.commission(0), "comm0")
    plot(strategy.closedtrades.commission(1), "comm1")
    plot(strategy.closedtrades.commission(2), "comm2")
    plot(strategy.closedtrades.profit(0), "profit0")
    plot(strategy.closedtrades.profit(1), "profit1")
    plot(strategy.closedtrades.profit(2), "profit2")


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
def __test_cash_per_order_is_charged_once_per_reversal__(script_path, module_key):
    """Four orders, four fees -- not six."""
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
    # Gross: +10 on the long, +5 on the short, +15 on the second long.
    # Fees: 10 on the entry, 10 per reversal, 10 on the close_all.
    assert last["netprofit"] == -10.0, last["netprofit"]

    # The entry pays a whole fee, the reversal that closed it half of one.
    assert last["comm0"] == 15.0, last["comm0"]
    # Opened by one reversal, closed by the next: half a fee on each side.
    assert last["comm1"] == 10.0, last["comm1"]
    # Opened by a reversal, closed by a whole-fee close_all.
    assert last["comm2"] == 15.0, last["comm2"]

    assert last["profit0"] == -5.0, last["profit0"]
    assert last["profit1"] == -5.0, last["profit1"]
    assert last["profit2"] == 0.0, last["profit2"]
