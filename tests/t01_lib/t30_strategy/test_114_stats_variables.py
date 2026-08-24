"""
@pyne

The strategy statistics variables that report a percentage or a peak.

Measured on BINANCE:BTCUSDT 30m with an initial capital of 1000000, pyramiding
3 and a 0.05% percent commission, long legs of 1 and 2 and short legs of 4 and
3. The reference covers the first 400 bars of that run; the laws it pins down:

* ``avg_*_trade_percent`` average the closed trades' OWN profit percentages --
  each one against that trade's entry cost including the entry fee -- not the
  currency averages against the initial capital.
* ``grossloss_percent`` counts the open commission, so a position that is still
  open already shows a loss percent, while ``grossprofit_percent`` does not.
* ``avg_losing_trade`` divides that same open-commission-inclusive gross loss and
  reports it POSITIVE -- 789.13114 = (602.310640000007 + 186.8205) / 1 while one
  short leg is open -- even though ``avg_losing_trade_percent`` is negative.
* ``max_contracts_held_*`` peak on the POSITION: the two long legs report 3 and
  the two short legs 7.
* ``max_drawdown_percent`` / ``max_runup_percent`` divide each excursion by its
  HIGHER endpoint -- the equity peak the drop fell from, the top the rise
  reached -- and are tracked separately from the currency maxima.
* ``closedtrades.first_index`` stays 0.
"""
from pynecore.lib import script, strategy, plot, bar_index


@script.strategy("Stats Variables", overlay=False, initial_capital=1000000,
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

    plot(strategy.grossprofit_percent, "gpp")
    plot(strategy.grossloss_percent, "glp")
    plot(strategy.avg_trade_percent, "atp")
    plot(strategy.avg_winning_trade_percent, "awtp")
    plot(strategy.avg_losing_trade_percent, "altp")
    plot(strategy.avg_losing_trade, "alt")
    plot(strategy.avg_winning_trade, "awt")
    plot(strategy.max_contracts_held_all, "mcha")
    plot(strategy.max_contracts_held_long, "mchl")
    plot(strategy.max_contracts_held_short, "mchs")
    plot(strategy.closedtrades.first_index, "fi")
    plot(strategy.max_drawdown, "md")
    plot(strategy.max_drawdown_percent, "mdp")
    plot(strategy.max_runup, "mr")
    plot(strategy.max_runup_percent, "mrp")


# noinspection PyShadowingNames
def __test_stats_variables__(csv_reader, runner, dict_comparator):
    """ Statistics variables follow the TradingView reference """
    with csv_reader('strategy_stats_vars.csv', subdir="data") as cr:
        r = runner(cr, syminfo_override=dict(prefix="BINANCE", ticker="BTCUSDT", currency="USDT",
                                             period="30", mintick=0.01, pricescale=100,
                                             mincontract=0.00001))
        bars = 0
        for candle, plot, _new_closed_trades in r.run_iter():
            dict_comparator(plot, candle.extra_fields)
            bars += 1
        assert bars == 400, bars
