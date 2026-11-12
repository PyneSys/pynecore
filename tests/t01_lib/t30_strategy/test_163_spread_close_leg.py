"""
@pyne

``strategy(spread_type=strategy.spread.trade)`` on the closing leg of the intrabar walk:
an exit whose entry filled mid-bar can become executable only because the spread widens
toward the close. It then fills where the closing leg starts, never outside the bar's
bid range.
"""
from pynecore.lib import bar_index, script, strategy


@script.strategy("Spread close leg", initial_capital=100000, default_qty_type=strategy.fixed,
                 default_qty_value=1, spread_type=strategy.spread.trade)
def main():
    if bar_index == 0:
        strategy.entry("S", strategy.short, stop=95.0)
        strategy.exit("XS", "S", stop=99.0)


def __test_helper_candles():
    from pynecore.types.ohlcv import OHLCV, AskOHLC

    flat = AskOHLC(100.0, 100.0, 100.0, 100.0)
    candles = [
        OHLCV(1_700_000_000_000, 100.0, 100.0, 100.0, 100.0, 100.0, ask=flat),
        # open -> high -> low -> close; no spread at the low, ten at the close
        OHLCV(1_700_003_600_000, 100.0, 101.0, 90.0, 100.0, 100.0,
              ask=AskOHLC(101.0, 111.0, 90.0, 110.0)),
    ]
    for index in range(2, 4):
        candles.append(OHLCV(1_700_000_000_000 + index * 3_600_000, 100.0, 100.0, 100.0, 100.0,
                             100.0, ask=flat))
    return candles


def __test_spread_close_leg_fills_inside_the_bar__(runner):
    """ A stop the widening close spread crosses fills at the closing leg's start """
    trades = {}
    r = runner(iter(__test_helper_candles()),
               syminfo_override=dict(mintick=0.01, minmove=1, pricescale=100))
    for _candle, _plot, new_closed_trades in r.run_iter():
        for trade in new_closed_trades:
            trades[trade.entry_id] = (round(trade.entry_price, 5), round(trade.exit_price, 5))

    # The sell stop entry fills on the descent; its buy stop is crossed at the bar's low,
    # where the bid is 90 and the close spread puts the ask at 100
    assert trades == {"S": (95.0, 100.0)}, trades
