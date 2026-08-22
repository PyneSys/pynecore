"""
A stop/limit level within 1e-4 of a tick is snapped ONTO that tick.

MEASURED on TradingView (BINANCE:BTCUSDT 30m, 66 short-stop and 55 long-stop
events, plus BINANCE:ADAUSDT at mintick 1e-4). A short's ``strategy.exit`` stop
rounds UP to the tick grid, a long's rounds DOWN -- but only once the level is
at least 1e-4 of a tick off the grid:

| direction | off-grid by  | snapped to      |
|-----------|--------------|-----------------|
| ceil      | 9.63e-5 tick | the tick itself |
| ceil      | 1.00e-4 tick | one tick up     |
| floor     | 9.00e-5 tick | the tick itself |
| floor     | 1.50e-4 tick | one tick down   |

The width is in TICKS, not in price units and not in ULPs: the same 1e-4
separates the two outcomes whether the tick count is 5e3 or 8.7e6. The trigger
for the measurement was a "Trend Surfers" stop at 86975.0500009212 (9.21e-5 of
a tick above the grid) that TradingView filled at 86975.05.
"""
from pynecore.lib import syminfo
# noinspection PyProtectedMember
from pynecore.lib.strategy import _price_round


_BASE = 86975.05  # exactly 8697505 ticks on a pricescale-100 grid
_TICK = 0.01  # 86975.06 / 86975.04 are the neighbouring grid points


def _with_grid(fn):
    """Run ``fn`` on a pricescale-100 / minmove-1 grid, then restore syminfo."""
    prev = (syminfo.pricescale, syminfo.minmove, syminfo.mintick)
    syminfo.pricescale, syminfo.minmove, syminfo.mintick = 100, 1, _TICK
    try:
        return fn()
    finally:
        syminfo.pricescale, syminfo.minmove, syminfo.mintick = prev


def __test_ceil_keeps_a_level_within_a_tick_ten_thousandth__():
    def check():
        assert _price_round(_BASE, 1) == _BASE
        assert _price_round(_BASE + 9.212e-7, 1) == _BASE
        assert _price_round(_BASE + 9.625e-7, 1) == _BASE
    _with_grid(check)


def __test_ceil_steps_up_from_one_tick_ten_thousandth__():
    def check():
        assert _price_round(_BASE + 1.004752e-6, 1) == 86975.06
        assert _price_round(_BASE + 1.5e-6, 1) == 86975.06
        assert _price_round(_BASE + 0.004, 1) == 86975.06
    _with_grid(check)


def __test_floor_keeps_a_level_within_a_tick_ten_thousandth__():
    def check():
        assert _price_round(_BASE, -1) == _BASE
        assert _price_round(_BASE - 9.0e-7, -1) == _BASE
    _with_grid(check)


def __test_floor_steps_down_from_one_tick_ten_thousandth__():
    def check():
        assert _price_round(_BASE - 1.5e-6, -1) == 86975.04
        assert _price_round(_BASE - 0.004, -1) == 86975.04
    _with_grid(check)


def __test_tolerance_is_in_ticks_not_in_price__():
    """A 1e-4 tick width on a 1e-4 grid is 1e-8 in price, and still steps up."""
    prev = (syminfo.pricescale, syminfo.minmove, syminfo.mintick)
    syminfo.pricescale, syminfo.minmove, syminfo.mintick = 10000, 1, 0.0001
    try:
        # 5000 ticks; 2e-8 price is 2e-4 of a tick, past the width
        assert _price_round(0.5 + 2e-8, 1) == 0.5001
        # 5e-9 price is 5e-5 of a tick, inside it
        assert _price_round(0.5 + 5e-9, 1) == 0.5
    finally:
        syminfo.pricescale, syminfo.minmove, syminfo.mintick = prev
