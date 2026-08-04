"""
@pyne

Order-type test strategy 3/6 — Long, stop entry above prior high

Buy-stop at high[1]; protective sell-stop at low[1].

Every position also closes via strategy.close() once open profit reaches
TAKE_PCT, provided the protective stop has not already fired. Entry and exit
comments carry the SIGNAL price; the trade CSV's own Price column carries the
actual FILL price, so comparing the two shows slippage and fill timing.
"""
from pynecore.lib import (
    script, strategy, close, high, low, plot, display,
)
from pynecore.lib import open as lib_open

#: close() once open profit reaches this fraction
TAKE_PCT = 0.02


# VN30F1M notional = price x 100,000 VND/point (~192.5m at 1925); DNSE's loan
# package quotes initialRate 0.1848 -> ~35.6m margin per contract. Capital must
# clear that or every entry is silently rejected on margin.
@script.strategy("T3 Long, stop entry above prior high", overlay=True, pyramiding=0,
                 initial_capital=500_000_000, default_qty_type=strategy.fixed,
                 default_qty_value=1, margin_long=18.48, margin_short=18.48,
                 calc_on_every_tick=False, process_orders_on_close=False)
def main():

    flat = strategy.position_size == 0
    in_long = strategy.position_size > 0

    if flat:
        strategy.entry("L", strategy.long, stop=high[1],
                       comment=f"E-stop@{high[1]}")

    if in_long:
        strategy.exit("X", from_entry="L", stop=low[1],
                      comment_loss=f"SL@{low[1]}")
        if close >= strategy.position_avg_price * (1 + TAKE_PCT):
            strategy.close("L", comment=f"TP@{close}")
    plot(strategy.position_size, "pos", display=display.data_window)
    plot(strategy.position_avg_price, "avg", display=display.data_window)
