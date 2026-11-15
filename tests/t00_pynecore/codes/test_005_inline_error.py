"""
@pyne

An inlined ``array.get`` that indexes past the end is R0010, as the call would
be: the transform copies the body into the call site, and the run finds the
call in the module's table of inlined calls.
"""
from pynecore.lib import array, bar_index, plot, script
from pynecore.types import Persistent


@script.indicator(title="Inline error")
def main():
    kept: Persistent[list[float]] = array.from_items(1.0, 2.0, 3.0)
    fresh = array.from_items(4.0, 5.0, 6.0)
    plot(array.get(fresh, 1) + array.get(kept, bar_index), "v")


def __test_helper_bars():
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=1_704_067_200_000 + i * 86_400_000, open=1.0, high=1.0,
                  low=1.0, close=1.0, volume=1.0) for i in range(6)]


def __test_an_inlined_array_get_out_of_range_is_r0010__(runner):
    """The index and the size come from the failing frame, the call's own line"""
    import pytest
    from pynecore import errorcodes

    with pytest.raises(errorcodes.IndexOutOfRangeError) as raised:
        for _ in runner(__test_helper_bars(), {"period": "1D"}).run_iter():
            pass
    error = raised.value
    assert isinstance(error, IndexError) and error.bar_index == 3
    assert error.params == {'function': 'array.get', 'index': 3, 'size': 3}
    assert error.at is not None
    assert (error.at.module, error.at.path, error.at.line) == ('main', __file__, 16)


def __test_the_module_records_its_inlined_calls__(runner):
    """One row per inlined call: its point, its Pine name and its argument holders"""
    module = runner(__test_helper_bars(), {"period": "1D"}).script_module
    rows = {(line, col): (function, holders)
            for line, col, function, holders in module.__pyne_inline_sites__}
    function, holders = rows[(16, 9)]
    assert function == 'array.get' and holders == (('n', 'fresh'), ('c', 1))
    function, holders = rows[(16, 31)]
    assert function == 'array.get' and holders[0][0] == 's' and holders[1][0] == 'n'
