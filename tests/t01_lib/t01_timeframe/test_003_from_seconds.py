"""
@pyne
"""
from pynecore.lib import script, timeframe


@script.indicator(title="Timeframe from_seconds()", shorttitle="tf_fs")
def main():
    # Test basic conversions
    assert timeframe.from_seconds(1) == "1S"
    assert timeframe.from_seconds(30) == "30S"
    assert timeframe.from_seconds(60) == "1"  # 1 minute - THIS WAS THE BUG!
    assert timeframe.from_seconds(120) == "2"  # 2 minutes
    assert timeframe.from_seconds(300) == "5"  # 5 minutes
    assert timeframe.from_seconds(900) == "15"  # 15 minutes
    assert timeframe.from_seconds(3600) == "60"  # 1 hour
    assert timeframe.from_seconds(14400) == "240"  # 4 hours
    assert timeframe.from_seconds(86400) == "1D"  # 1 day
    assert timeframe.from_seconds(172800) == "2D"  # 2 days
    assert timeframe.from_seconds(604800) == "1W"  # 1 week
    assert timeframe.from_seconds(1209600) == "2W"  # 2 weeks
    assert timeframe.from_seconds(2419200) == "1M"  # 1 month (4 weeks)
    assert timeframe.from_seconds(4838400) == "2M"  # 2 months
    
    # Edge cases
    assert timeframe.from_seconds(59) == "59S"  # Not divisible by 60
    assert timeframe.from_seconds(61) == "61S"  # Not divisible by 60
    assert timeframe.from_seconds(180) == "3"  # 3 minutes
    assert timeframe.from_seconds(240) == "4"  # 4 minutes


def __test_timeframe_from_seconds__(runner, dummy_ohlcv_iter):
    """ timeframe.from_seconds() """
    next(runner(dummy_ohlcv_iter).run_iter())