from zoneinfo import ZoneInfo

from pynecore import lib
from pynecore.core.script_runner import _set_lib_properties
from pynecore.types.ohlcv import OHLCV


def __test_set_lib_properties_keeps_historical_last_bar_index_fixed__():
    """ Historical last_bar_index stays fixed at the final bar """
    candle = OHLCV(
        timestamp=1735689600,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
    )

    _set_lib_properties(candle, 2, ZoneInfo("UTC"), lib, None, last_bar_index=4)

    assert lib.bar_index == 2
    assert lib.last_bar_index == 4
