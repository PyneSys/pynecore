"""
@pyne

A failed ``assert`` of the script stops the run as R0004 on its bar, with the
assertion's message.
"""
from pynecore.lib import bar_index, close, plot, script


@script.indicator(title="Script assert")
def main():
    plot(close, "close")
    assert bar_index < 4, "the bar index reached 4"


def __test_a_failed_assert_of_the_script_stops_the_run_as_r0004__(runner):
    """The script's own ``assert`` is R0004 on its bar, located at its line"""
    import pytest
    from pynecore import errorcodes
    from pynecore.types.ohlcv import OHLCV

    bars = [OHLCV(timestamp=1_704_067_200_000 + i * 86_400_000, open=1.0, high=1.0,
                  low=1.0, close=1.0, volume=1.0) for i in range(6)]
    with pytest.raises(errorcodes.AssertionFailedError) as raised:
        for _ in runner(bars, {"period": "1D"}).run_iter():
            pass
    error = raised.value
    assert isinstance(error, AssertionError) and error.code is errorcodes.ASSERTION_FAILED
    assert error.params == {'message': 'the bar index reached 4'}
    assert error.bar_index == 4
    assert error.at is not None and error.at.path == __file__ and error.at.line == 13
    assert error.report().startswith("bar 4 (2024-01-05 00:00 UTC): R0004 at ")
    assert error.report().endswith(": Assertion failed. the bar index reached 4")
