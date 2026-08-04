"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, log, bar_index, ta


@script.indicator(title="Cross Adjacent", shorttitle="cross_adjacent")
def main():
    # The source flips sign on every bar, so a crossover and a crossunder land
    # on consecutive bars -- the case where ``ta.cross`` must keep BOTH halves'
    # previous-bar relation up to date. Expected values measured on TradingView
    # (BINANCE:BTCUSDT 30m, probe run 2026-08-04): cross is true on every bar,
    # alternating between the over and the under leg.
    a: Series[float] = 1.0 if bar_index % 2 == 0 else -1.0
    b = 0.0

    crossed = ta.cross(a, b)
    crossed_over = ta.crossover(a, b)
    crossed_under = ta.crossunder(a, b)

    if 3 <= bar_index <= 12:
        log.info("a={0} cross={1} over={2} under={3}",
                 a, crossed, crossed_over, crossed_under)


def __test_cross_adjacent__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Cross on consecutive bars """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(13):
            next(run_iter)
