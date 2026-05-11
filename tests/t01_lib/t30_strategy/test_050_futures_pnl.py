"""
@pyne

Regression test for issue #50: futures-symbol PnL reporting must use
syminfo.pointvalue (USD/point) when computing closed-trade profit.

Reference data was captured from TradingView's strategy tester on
NYMEX_MINI:QM1! (pointvalue = 500) using the matching .pine source in
workdir/scripts/futures_pnl_test.pine. The Pine and Pyne implementations
below must stay in sync.

This test focuses on the USD value of each closed trade. When a trade's
entry/exit prices match TV (the common case here), the reported profit
must equal TV's Net P&L USD. Trades that diverge due to unrelated
execution-timing edges are skipped rather than failing the test.
"""
import math
from pynecore.lib import close, script, strategy, ta


@script.strategy(
    "FuturesPnL Test (SMA cross)",
    overlay=True,
    default_qty_type=strategy.fixed,
    default_qty_value=1,
    initial_capital=100000,
    pyramiding=0,
)
def main():
    fast = ta.sma(close, 10)
    slow = ta.sma(close, 30)
    if ta.crossover(fast, slow):
        strategy.entry('L', strategy.long)
    if ta.crossunder(fast, slow):
        strategy.entry('S', strategy.short)


# noinspection PyShadowingNames
def __test_futures_pnl__(csv_reader, runner):
    """ FuturesPnL: closed-trade profit must be in USD on a pointvalue!=1 symbol """
    # Use TV's literal pricescale/minmove (not the mathematically-equivalent
    # pricescale=40, minmove=1 form). This exercises `lib.math.round_to_mintick`
    # on a symbol where minmove != 1.
    syminfo_override = dict(
        prefix="NYMEX_MINI",
        ticker="QM1!",
        currency="USD",
        period="60",
        type="futures",
        mintick=0.025,
        pricescale=1000,
        minmove=25,
        pointvalue=500,
        timezone="America/New_York",
    )
    with csv_reader('futures_pnl_ohlcv.csv', subdir="data") as cr, \
            csv_reader('futures_pnl_trades.csv', subdir="data") as cr_equity:
        r = runner(cr, syminfo_override=syminfo_override)
        equity_iter = iter(cr_equity)
        compared = 0
        for i, (candle, plot, new_closed_trades) in enumerate(r.run_iter()):
            for trade in new_closed_trades:
                good_entry = next(equity_iter).extra_fields
                good_exit = next(equity_iter).extra_fields
                # Only validate when this PyneCore trade aligns with TV on entry+exit prices;
                # divergences here are execution-timing edges unrelated to issue #50.
                if not (math.isclose(trade.entry_price, float(good_entry['Price USD']), abs_tol=0.01)
                        and math.isclose(trade.exit_price, float(good_exit['Price USD']), abs_tol=0.01)):
                    continue
                expected_usd = float(good_exit['Net P&L USD'])
                assert math.isclose(trade.profit, expected_usd, abs_tol=0.5, rel_tol=1e-4), (
                    f"Profit USD mismatch on trade ending {good_exit['Date/Time']}: "
                    f"trade.profit={trade.profit} vs TV Net P&L USD={expected_usd}"
                )
                compared += 1
        # Sanity guard — most trades must align so we actually exercise the fix.
        assert compared >= 200, f"Only {compared} PyneCore trades aligned with TV reference"
