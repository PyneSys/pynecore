"""
@pyne

A closed trade's own percentages and its excursion extremes.

Measured on BINANCE:BTCUSDT 30m (initial capital 1000000, pyramiding 3, 0.05%
percent commission, long legs of 1 and 2, short legs of 4 and 3), first 400
bars. Two laws the reference pins down:

* Every percentage divides by the trade's TOTAL ENTRY COST -- position value
  plus the fee paid to open it. Against the bare position value each percent
  comes out 5e-4 relative too high at this commission.
* ``max_drawdown`` / ``max_runup`` are measured from the trade's OWN entry
  price, not from the position average a later leg shifted: trade 0 (long 1 at
  93761.90000000001, worst low 93500.0, best high 94509.42) reports
  308.78095 = (93761.9 - 93500.0) + 46.88095 and 700.63905 =
  (94509.42 - 93761.9) - 46.88095.
"""
from pynecore.lib import script, strategy, plot, bar_index


@script.strategy("Trade Percents", overlay=False, initial_capital=1000000,
                 default_qty_type=strategy.fixed, default_qty_value=1, pyramiding=3,
                 commission_type=strategy.commission.percent, commission_value=0.05)
def main():
    c = bar_index % 40
    if c == 0:
        strategy.entry("L1", strategy.long, qty=1)
    if c == 5:
        strategy.entry("L2", strategy.long, qty=2)
    if c == 10:
        strategy.close_all()
    if c == 20:
        strategy.entry("S1", strategy.short, qty=4)
    if c == 25:
        strategy.entry("S2", strategy.short, qty=3)
    if c == 30:
        strategy.close_all()

    plot(strategy.closedtrades.profit_percent(0), "pp0")
    plot(strategy.closedtrades.profit_percent(1), "pp1")
    plot(strategy.closedtrades.max_drawdown(0), "md0")
    plot(strategy.closedtrades.max_drawdown_percent(0), "mdp0")
    plot(strategy.closedtrades.max_runup(0), "mr0")
    plot(strategy.closedtrades.max_runup_percent(0), "mrp0")
    plot(strategy.closedtrades.max_drawdown(1), "md1")
    plot(strategy.closedtrades.max_drawdown_percent(1), "mdp1")
    plot(strategy.closedtrades.max_runup(1), "mr1")
    plot(strategy.closedtrades.max_runup_percent(1), "mrp1")


# noinspection PyShadowingNames
def __test_trade_percents__(csv_reader, runner, dict_comparator):
    """ Closed-trade percentages follow the TradingView reference """
    with csv_reader('strategy_trade_percents.csv', subdir="data") as cr:
        r = runner(cr, syminfo_override=dict(prefix="BINANCE", ticker="BTCUSDT", currency="USDT",
                                             period="30", mintick=0.01, pricescale=100,
                                             mincontract=0.00001))
        bars = 0
        for candle, plot, _new_closed_trades in r.run_iter():
            dict_comparator(plot, candle.extra_fields)
            bars += 1
        assert bars == 400, bars
