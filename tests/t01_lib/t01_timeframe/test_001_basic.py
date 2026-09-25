"""
@pyne
"""
from pynecore.lib import script, timeframe, syminfo


# PyCharm keeps each asserted timeframe.* value narrowed across the syminfo.period changes,
# so a later block asserting another value reads as unreachable
# noinspection PyUnreachableCode
@script.indicator(title="Timeframe Basic", shorttitle="tf_basic")
def main():
    import pytest

    # No script-level ``timeframe``: every ``timeframe.*`` builtin describes the
    # chart, so they all follow ``syminfo.period`` below. A script that DOES declare
    # one reports THAT timeframe instead -- MEASURED on TradingView
    # (CAPITALCOM:GOLD@60 with ``timeframe='W'``: period, main_period, isweekly and
    # multiplier all answer for the weekly context) and covered by
    # ``tests/t00_pynecore/core/test_100_script_timeframe_gaps.py``.
    assert timeframe.main_period == "5"

    syminfo.period = "10T"
    assert timeframe.multiplier == 10
    assert timeframe.period == "10T"
    assert timeframe.isdaily is False
    assert timeframe.isdwm is False
    assert timeframe.isintraday is True
    assert timeframe.isminutes is False
    assert timeframe.ismonthly is False
    assert timeframe.isseconds is False
    assert timeframe.isticks is True
    assert timeframe.isweekly is False

    syminfo.period = "30S"
    assert timeframe.multiplier == 30
    assert timeframe.period == "30S"
    assert timeframe.isdaily is False
    assert timeframe.isdwm is False
    assert timeframe.isintraday is True
    assert timeframe.isminutes is False
    assert timeframe.ismonthly is False
    assert timeframe.isseconds is True
    assert timeframe.isticks is False
    assert timeframe.isweekly is False

    syminfo.period = "5"
    assert timeframe.multiplier == 5
    assert timeframe.period == "5"
    assert timeframe.isdaily is False
    assert timeframe.isdwm is False
    assert timeframe.isintraday is True
    assert timeframe.isminutes is True
    assert timeframe.ismonthly is False
    assert timeframe.isseconds is False
    assert timeframe.isticks is False
    assert timeframe.isweekly is False

    syminfo.period = "D"
    assert timeframe.multiplier == 1
    assert timeframe.period == "D"
    assert timeframe.isdaily is True
    assert timeframe.isdwm is True
    assert timeframe.isintraday is False
    assert timeframe.isminutes is False
    assert timeframe.ismonthly is False
    assert timeframe.isseconds is False
    assert timeframe.isticks is False
    assert timeframe.isweekly is False
    syminfo.period = "4D"
    assert timeframe.multiplier == 4
    assert timeframe.period == "4D"

    syminfo.period = "5W"
    assert timeframe.multiplier == 5
    assert timeframe.period == "5W"
    assert timeframe.isdaily is False
    assert timeframe.isdwm is True
    assert timeframe.isintraday is False
    assert timeframe.isminutes is False
    assert timeframe.ismonthly is False
    assert timeframe.isseconds is False
    assert timeframe.isticks is False
    assert timeframe.isweekly is True

    syminfo.period = "1M"
    assert timeframe.multiplier == 1
    assert timeframe.period == "1M"
    assert timeframe.isdaily is False
    assert timeframe.isdwm is True
    assert timeframe.isintraday is False
    assert timeframe.isminutes is False
    assert timeframe.ismonthly is True
    assert timeframe.isseconds is False
    assert timeframe.isticks is False
    assert timeframe.isweekly is False
    syminfo.period = "3M"
    assert timeframe.multiplier == 3
    assert timeframe.period == "3M"

    with pytest.raises(AssertionError):
        syminfo.period = "3Y"
        assert timeframe.multiplier == 1

    with pytest.raises(AssertionError):
        syminfo.period = "5X"
        assert timeframe.change("1D")


def __test_timeframe_basic__(runner, dummy_ohlcv_iter):
    """ Basic """
    next(runner(dummy_ohlcv_iter).run_iter())
