"""
@pyne

A series parameter whose default is dynamic records its VALUE, not the sentinel.

``DynamicDefaultTransformer`` replaces a default that reads per-bar state with
a sentinel and resolves it in a guard at the top of the body. The series pass
prepends its own parameter recording, which has to land after that guard --
above it the sentinel object itself went into the series, and an int or float
series (which stores every value as a float) raised on it.

The guard block is contiguous but not necessarily first: an earlier pass may
have prepended a hoisted lib-series declaration of its own, which is what makes
"skip the leading guards" too narrow a rule.
"""
from pynecore.lib import script, time, close, na
from pynecore.types.series import Series


@script.indicator("series param dynamic default")
def main():
    def observe(stamp: Series[int] = time, price: Series[float] = close):
        # The lib series read here is hoisted to the very top of the body,
        # ahead of the sentinel guards
        first = na(stamp[1]) or stamp[1] < time
        return stamp, price, first

    stamp, price, first = observe()
    given_stamp, given_price, _ = observe(1_704_067_200_000, 42.0)
    return {
        "stamp": stamp,
        "price": price,
        "monotonic": 1.0 if first else 0.0,
        "given_stamp": given_stamp,
        "given_price": given_price,
    }


def __test_a_dynamic_default_reaches_the_series_as_a_value__(csv_reader, runner):
    """The recorded value is the resolved default, never the sentinel"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            assert type(plot["stamp"]) is float
            assert plot["stamp"] == candle.timestamp
            assert type(plot["price"]) is float
            assert plot["price"] == candle.close
            assert plot["monotonic"] == 1.0
            # An explicitly passed argument is unaffected
            assert plot["given_stamp"] == 1_704_067_200_000
            assert plot["given_price"] == 42.0
            if i >= 2:
                break
