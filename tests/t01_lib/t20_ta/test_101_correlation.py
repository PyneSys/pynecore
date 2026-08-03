"""
@pyne
"""
from pynecore.lib import script, ta, close


@script.indicator(title="Correlation Test", shorttitle="Corr_T", overlay=False)
def main():
    return {
        # This also tests the dev() function
        "correlation": ta.correlation(ta.sma(close, 20), ta.ema(close, 10), 20),
    }


def __test_correlation__(csv_reader, runner, dict_comparator, log):
    """ Correlation """
    with csv_reader('correlation.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            dict_comparator(plot, candle.extra_fields)
            if i > 150:
                break
