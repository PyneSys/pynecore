"""
@pyne
"""
from pynecore.lib import script, log, bar_index, ta, na


@script.indicator(title="WMA Exact", shorttitle="wma_exact")
def main():
    # TradingView parity for the weighted moving average. Every reference below is
    # the full-precision value TradingView produced for these exact series (probe
    # m560, 2026-08-03); the log rounds to three decimals, so each value is
    # amplified against its reference, where one ulp is ~14 units.
    a = 100.0 + (bar_index % 7) / 3.0
    # Unlike the other rolling machines wma does not compact the window: an na bar
    # carries the previous value forward and only the bar itself returns na.
    g = na if bar_index % 3 == 0 else a

    k1 = ta.wma(a, 3)
    k2 = ta.wma(a, 5)
    k3 = ta.wma(g, 4)
    k4 = ta.wma(a, 1)
    # hma is three nested wma calls, so it pins the machine a second time.
    k5 = ta.hma(a, 4)

    r1 = 100.77777777777779 if bar_index == 10 else 101.11111111111113
    r2 = 100.71111111111112 if bar_index == 10 else 100.88888888888889
    r3 = 100.56666666666666 if bar_index == 10 else 100.93333333333332
    r4 = 101.0 if bar_index == 10 else 101.33333333333333
    r5 = 100.92222222222223 if bar_index == 10 else 101.33333333333333

    if bar_index == 10 or bar_index == 11:
        log.info("{0}|{1}|{2}|{3}|{4}", (k1 - r1) * 1e15, (k2 - r2) * 1e15,
                 (k3 - r3) * 1e15, (k4 - r4) * 1e15, (k5 - r5) * 1e15)


def __test_wma_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ WMA TradingView-exact machine """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(14):
            next(run_iter)
