"""
@pyne

Order-type test strategy 4/6 — Short, stop entry below prior low

Sell-stop at low[1]; protective buy-stop at high[1].

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
@script.strategy("T4 Short, stop entry below prior low", overlay=True, pyramiding=0,
                 initial_capital=500_000_000, default_qty_type=strategy.fixed,
                 default_qty_value=1, margin_long=18.48, margin_short=18.48,
                 calc_on_every_tick=False, process_orders_on_close=False)
def main():

    flat = strategy.position_size == 0
    in_short = strategy.position_size < 0

    if flat:
        strategy.entry("S", strategy.short, stop=low[1],
                       comment=f"E-stop@{low[1]}")

    if in_short:
        strategy.exit("X", from_entry="S", stop=high[1],
                      comment_loss=f"SL@{high[1]}")
        if close <= strategy.position_avg_price * (1 - TAKE_PCT):
            strategy.close("S", comment=f"TP@{close}")
    plot(strategy.position_size, "pos", display=display.data_window)
    plot(strategy.position_avg_price, "avg", display=display.data_window)
