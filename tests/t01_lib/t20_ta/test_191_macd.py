"""
@pyne
"""
from pynecore.lib import script, ta, plot, close, color


@script.indicator(title="MACD Test", shorttitle="macd", overlay=False)
def main():
    macd_val, macd_signal, macd_hist = ta.macd(close, 12, 26, 9)
    plot(macd_val, "ta.macd(close, 12, 26, 9).macd", color=color.blue)
    plot(macd_signal, "ta.macd(close, 12, 26, 9).signal", color=color.orange)
    plot(macd_hist, "ta.macd(close, 12, 26, 9).hist", color=color.gray)


# noinspection PyShadowingNames
def __test_macd__(csv_reader, runner, dict_comparator, log):
    """ MACD """
    with csv_reader('macd.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            dict_comparator(plot, candle.extra_fields)  # I cannot make it more precise
            if i > 200:
                break
