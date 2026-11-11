"""
@pyne

``strategy(spread_type=strategy.spread.trade)``: a buy fills at the ask, the bid-side
price plus the bar's spread at the node the fill happens on; a sell fills at the bid.
Limit fills pay no spread, like they pay no slippage. A bar without a quoted ask
charges ``syminfo.avg_spread``.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy("Spread trade", initial_capital=100000, default_qty_type=strategy.fixed,
                 default_qty_value=1, spread_type=strategy.spread.trade)
def main():
    if bar_index == 0:
        strategy.entry("L", strategy.long)
    if bar_index == 2:
        strategy.close("L")
    if bar_index == 4:
        strategy.entry("S", strategy.short)
    if bar_index == 6:
        strategy.close("S")
    if bar_index == 8:
        strategy.entry("LS", strategy.long, stop=1.1050)
    if bar_index == 10:
        strategy.close("LS")
    if bar_index == 12:
        strategy.entry("LL", strategy.long, limit=1.0950)
    if bar_index == 14:
        strategy.close("LL")


def __test_helper_candles():
    from pynecore.types.ohlcv import OHLCV, AskOHLC

    candles = []
    for index in range(17):
        high = 1.1060 if index == 9 else 1.1010
        low = 1.0940 if index == 13 else 1.0990
        # Spread: 2 pips at the open and the close, 3 pips at the extremes
        ask = AskOHLC(1.1002, high + 0.0003, low + 0.0003, 1.1002)
        # Bar 7 has no quoted ask, so the short's closing buy pays the average spread
        candles.append(OHLCV(1_700_000_000_000 + index * 3_600_000, 1.1000, high, low, 1.1000,
                             100.0, ask=None if index == 7 else ask))
    return candles


def __test_spread_trade_buys_at_the_ask__(runner):
    """ Buys pay the spread of their fill node, sells and limit fills do not """
    trades = {}
    r = runner(iter(__test_helper_candles()),
               syminfo_override=dict(mintick=0.0001, minmove=1, pricescale=10000, avg_spread=0.0005))
    for _candle, _plot, new_closed_trades in r.run_iter():
        for trade in new_closed_trades:
            trades[trade.entry_id] = (round(trade.entry_price, 5), round(trade.exit_price, 5))

    assert len(trades) == 4, trades
    # Market buy at the open pays the open spread; the closing sell fills at the bid
    assert trades["L"] == (1.1002, 1.1000)
    # Short: the sell fills at the bid, the closing buy on an unquoted bar pays avg_spread
    assert trades["S"] == (1.1000, 1.1005)
    # A buy stop fills at its level plus the spread at the extreme it triggered on
    assert trades["LS"] == (1.1053, 1.1000)
    # A buy limit fills at its limit price
    assert trades["LL"] == (1.0950, 1.1000)
