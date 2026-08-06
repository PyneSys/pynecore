"""
@pyne

Regression: request.seed keeps the arity of a tuple expression.

Seed repositories are unavailable to PyneCore, so the call returns ``na`` -- but a
Pine tuple expression yields a tuple, and a compiled script destructures it. One
``na`` for such a call halted the script with "na is not iterable" before the
``na()`` guards could ever run.
"""
from pynecore.lib import close, plot, request, script, ta


@script.indicator("seed tuple", "seedt")
def main():
    dev_act, dev_sma = request.seed("seed_crypto_santiment", "BTC_DEV_ACTIVITY",
                                    (close, ta.sma(close, 10)))
    single = request.seed("seed_crypto_santiment", "BTC_DEV_ACTIVITY", close)
    plot(dev_act, "DevAct")
    plot(dev_sma, "DevSMA")
    plot(single, "Single")


def __test_tuple_destructuring_yields_na_values__(runner):
    """The compiled ``a, b = request.seed(...)`` form runs and hands out na"""
    # Verified on TradingView (BINANCE:BTCUSDT 1D): with a seed series that has no
    # data both tuple members come back as na -- the tuple shape survives the miss
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV

    base = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [OHLCV(timestamp=base + i * 300_000, open=100.0, high=101.0, low=99.0,
                  close=100.5, volume=100.0) for i in range(3)]

    r = runner(iter(bars))
    for _, plot_values in r.run_iter():
        assert isinstance(plot_values['DevAct'], NA)
        assert isinstance(plot_values['DevSMA'], NA)
        assert isinstance(plot_values['Single'], NA)


def __test_expression_shape_decides_the_return_shape__():
    """A tuple keeps its arity, everything else stays a single na"""
    # PyneComp compiles a Pine tuple expression to a tuple literal, while an array
    # variable stays a list. Measured on TradingView: an array expression is legal
    # and returns a single na array id -- array.size() on it halts the script with
    # RE10052 -- so splitting a list into per-element na values would invent a
    # result TradingView never produces.
    from pynecore.types.na import NA

    assert isinstance(request.seed("seed_crypto_santiment", "BTC_DEV_ACTIVITY", 1.0), NA)
    assert isinstance(request.seed("seed_crypto_santiment", "BTC_DEV_ACTIVITY",
                                   [1.0, 2.0, 3.0]), NA)

    result = request.seed("seed_crypto_santiment", "BTC_DEV_ACTIVITY", (1.0, 2.0, 3.0))
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(item, NA) for item in result)


def __test_pine_argument_names__():
    """The Pine keywords address the parameters they name"""
    result = request.seed(source="seed_crypto_santiment", symbol="BTC_DEV_ACTIVITY",
                          expression=(1.0, 2.0), ignore_invalid_symbol=True,
                          calc_bars_count=100)
    assert isinstance(result, tuple)
    assert len(result) == 2
