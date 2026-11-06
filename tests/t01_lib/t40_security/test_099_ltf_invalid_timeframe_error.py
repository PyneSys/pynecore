"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script, syminfo


@script.indicator(title="LTF Invalid Timeframe Error", shorttitle="LTFITE")
def main():
    # Above the 60-minute chart, without ignore_invalid_timeframe
    c_d = request.security_lower_tf(syminfo.tickerid, "D", close)
    plot(array.size(c_d), "size")


def __test_an_ltf_above_the_script_timeframe_stops_the_run__(runner):
    """Without ``ignore_invalid_timeframe`` the run stops before its first bar.

    TradingView reports the error at the script's start; no bar is evaluated.
    """
    import pytest
    from pynecore.types.ohlcv import OHLCV

    bars = [OHLCV(timestamp=1_735_689_600_000 + i * 3_600_000, open=100.0, high=101.0,
                  low=99.0, close=100.0, volume=1.0) for i in range(48)]
    r = runner(bars, {"period": "60"})
    evaluated = []
    with pytest.raises(RuntimeError, match=r"'D' timeframe, which is higher than the '60'"):
        for candle, _values in r.run_iter():
            evaluated.append(candle)
    assert not evaluated, f"{len(evaluated)} bars ran before the error"
