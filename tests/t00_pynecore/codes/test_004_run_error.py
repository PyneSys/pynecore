"""
@pyne

A run stops on a coded error with the bar it stopped on and the script's line.
"""
from pynecore.lib import bar_index, close, plot, runtime, script


@script.indicator(title="Run error")
def main():
    if bar_index == 3:
        runtime.error("stopped at three")
    plot(close, "close")


def __test_a_run_stops_on_a_coded_error_at_its_bar_and_line__(runner):
    """``runtime.error()`` is R0005 on its bar, located at the calling line"""
    import pytest
    from pynecore import errorcodes
    from pynecore.types.ohlcv import OHLCV

    bars = [OHLCV(timestamp=1_704_067_200_000 + i * 86_400_000, open=1.0, high=1.0,
                  low=1.0, close=1.0, volume=1.0) for i in range(6)]
    with pytest.raises(errorcodes.ScriptRuntimeError) as raised:
        for _ in runner(bars, {"period": "1D"}).run_iter():
            pass
    error = raised.value
    assert isinstance(error, RuntimeError) and error.code is errorcodes.RUNTIME_ERROR
    assert (error.bar_index, error.bar_time) == (3, 1_704_067_200_000 + 3 * 86_400_000)
    assert error.at is not None and error.at.path == __file__ and error.at.line == 12
    assert error.at.module == 'main'
    assert error.report() == f"bar 3 (2024-01-04 00:00 UTC): R0005 at {__file__}:12:9: " \
                             "stopped at three"
