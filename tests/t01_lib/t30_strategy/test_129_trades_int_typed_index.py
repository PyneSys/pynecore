"""
@pyne

The trade accessors take an int-TYPED trade number.

Pine's ``int`` is a static type only, so ``9 / 4`` is int-typed with the value
2.25 and TradingView reads trade 2 with it. All 31 accessors (18 in
``closedtrades``, 13 in ``opentrades``) handed the number to
``closed_trades[trade_num]`` bare, which raised ``TypeError: sequence index must
be integer``. An ``na`` trade number was unhandled the same way.

MEASURED on TradingView (FX:EURUSD@60, ``d = (R + z) / 8`` = 1.75):
``strategy.closedtrades.entry_price(d)`` is 1.06042 and
``strategy.closedtrades.size(d)`` is 1 -- the values of trade 1.

The pairing below needs no reference data: a fractional trade number must
answer exactly what its truncated integer answers, on every bar.
"""
from pynecore.lib import script, strategy, plot, bar_index, close
from pynecore.types.na import NA


@script.strategy("Int-typed trade index", overlay=False,
                 default_qty_type=strategy.fixed, default_qty_value=1)
def main():
    c = bar_index % 20
    if c == 0:
        strategy.entry("L", strategy.long, qty=1)
    if c == 10:
        strategy.close_all()

    frac = 9 / 4  # int-typed, 2.25 -- trade 2
    open_frac = 1 / 2  # int-typed, 0.5 -- trade 0

    plot(strategy.closedtrades.entry_price(frac), "ep_frac")
    plot(strategy.closedtrades.entry_price(2), "ep_int")
    plot(strategy.closedtrades.exit_price(frac), "xp_frac")
    plot(strategy.closedtrades.exit_price(2), "xp_int")
    plot(strategy.closedtrades.size(frac), "sz_frac")
    plot(strategy.closedtrades.size(2), "sz_int")
    plot(strategy.closedtrades.profit(frac), "pf_frac")
    plot(strategy.closedtrades.profit(2), "pf_int")
    plot(strategy.closedtrades.entry_bar_index(frac), "ebi_frac")
    plot(strategy.closedtrades.entry_bar_index(2), "ebi_int")
    plot(strategy.closedtrades.commission(frac), "cm_frac")
    plot(strategy.closedtrades.commission(2), "cm_int")
    plot(strategy.opentrades.entry_price(open_frac), "oep_frac")
    plot(strategy.opentrades.entry_price(0), "oep_int")
    plot(strategy.opentrades.size(open_frac), "osz_frac")
    plot(strategy.opentrades.size(0), "osz_int")

    # An na trade number must answer like a missing trade instead of reaching
    # the subscript -- each accessor has its own "no such trade" answer
    na_num = NA(int)
    plot(strategy.closedtrades.entry_price(na_num), "epn_frac")
    plot(strategy.closedtrades.entry_price(-1), "epn_int")
    plot(strategy.opentrades.size(na_num), "oszn_frac")
    plot(strategy.opentrades.size(-1), "oszn_int")
    plot(close, "close")


def __test_trades_int_typed_index__(runner):
    """A fractional trade number reads the trade its truncation names"""
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    seed = 7717
    price = 100.0
    rows = []
    for bar in range(120):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        price += (seed / 0x7FFFFFFF - 0.5) * 4.0
        rows.append(OHLCV(timestamp=base_ts + bar * 1800, open=price, high=price + 1.5,
                          low=price - 1.5, close=price, volume=10.0))

    def is_na(value):
        return isinstance(value, NA) or value != value

    pairs = ("ep", "xp", "sz", "pf", "ebi", "cm", "oep", "osz", "epn", "oszn")
    compared = 0
    for i, (_candle, plot_values, _closed) in enumerate(runner(iter(rows)).run_iter()):
        for name in pairs:
            frac, exact = plot_values[f"{name}_frac"], plot_values[f"{name}_int"]
            if is_na(frac) or is_na(exact):
                assert is_na(frac) and is_na(exact), \
                    f"{name} na-disagrees at bar {i}: {frac} vs {exact}"
                continue
            assert frac == exact, f"{name} differs at bar {i}: {frac} vs {exact}"
            compared += 1

    assert compared > 200, f"too few non-na comparisons: {compared}"
