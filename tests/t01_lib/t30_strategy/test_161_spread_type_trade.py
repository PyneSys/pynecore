"""
@pyne

``strategy(spread_type=strategy.spread.trade)``: a buy fills at the ask, a sell at the
bid, and every order is triggered by the side it trades on: buy limits and buy stops
by the ask, sell limits and sell stops by the bid. A bar without a quoted ask
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
        strategy.entry("LS", strategy.long, stop=1.1012)
    if bar_index == 10:
        strategy.cancel("LS")
        strategy.close("LS")
    if bar_index == 12:
        strategy.entry("LL", strategy.long, limit=1.0990)
    if bar_index == 14:
        strategy.cancel("LL")
    if bar_index == 15:
        strategy.entry("LL2", strategy.long, limit=1.0992)
    if bar_index == 17:
        strategy.close("LL2")
    if bar_index == 19:
        strategy.entry("L2", strategy.long)
        strategy.exit("X2", "L2", stop=1.0990)
    if bar_index == 24:
        strategy.entry("S2", strategy.short)
        strategy.exit("XS2", "S2", stop=1.1010)
    if bar_index == 29:
        strategy.entry("S3", strategy.short)
        strategy.exit("XS3", "S3", trail_price=1.0986, trail_offset=5)
    if bar_index == 33:
        strategy.close("S3")


def __test_helper_candles():
    from pynecore.types.ohlcv import OHLCV, AskOHLC

    highs = {9: 1.1011, 26: 1.1009, 27: 1.1011}
    lows = {13: 1.0989, 16: 1.0989, 21: 1.0989, 22: 1.0987, 31: 1.0985}
    candles = []
    for index in range(36):
        high = highs.get(index, 1.1005)
        low = lows.get(index, 1.0995)
        # The ask is two pips above the bid all along the bar
        ask = AskOHLC(1.1002, high + 0.0002, low + 0.0002, 1.1002)
        # Bar 7 has no quoted ask, so a buy filling there pays the average spread
        candles.append(OHLCV(1_700_000_000_000 + index * 3_600_000, 1.1000, high, low, 1.1000,
                             100.0, ask=None if index == 7 else ask))
    return candles


def __test_helper_trades(runner) -> dict[str, tuple[float, float]]:
    trades = {}
    r = runner(iter(__test_helper_candles()),
               syminfo_override=dict(mintick=0.0001, minmove=1, pricescale=10000, avg_spread=0.0005))
    for _candle, _plot, new_closed_trades in r.run_iter():
        for trade in new_closed_trades:
            trades[trade.entry_id] = (round(trade.entry_price, 5), round(trade.exit_price, 5))
    return trades


def __test_spread_trade_triggers_on_the_trading_side__(runner):
    """ Buy orders are triggered by the ask and fill at their own level """
    trades = __test_helper_trades(runner)

    # Market buy at the open pays the open spread; the closing sell fills at the bid
    assert trades["L"] == (1.1002, 1.1000)
    # Short: the sell fills at the bid, the closing buy on an unquoted bar pays avg_spread
    assert trades["S"] == (1.1000, 1.1005)
    # The ask reaches the buy stop although the bid does not, and it fills at its level
    assert trades["LS"] == (1.1012, 1.1000)
    # The bid reaches the buy limit, the ask does not: it never fills
    assert "LL" not in trades
    # The ask reaches this one, and it fills at its limit
    assert trades["LL2"] == (1.0992, 1.1000)
    # A long's stop loss is triggered by the bid and fills at its level
    assert trades["L2"] == (1.1002, 1.0990)
    # A short's stop loss is triggered by the ask and fills at its level
    assert trades["S2"] == (1.1000, 1.1010)
    # The short's trailing stop follows the ask, which never reaches the activation level
    assert trades["S3"] == (1.1000, 1.1002)
    assert len(trades) == 7, trades
