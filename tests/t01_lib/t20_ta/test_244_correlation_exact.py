"""
@pyne
"""
from pynecore.lib import script, log, bar_index, ta, na


@script.indicator(title="Correlation Exact", shorttitle="correlation_exact")
def main():
    # TradingView parity for the correlation machine. Every expected value below
    # is TradingView's own log output for these exact series (probe m557,
    # 2026-08-03). The two non-trivial results are amplified against their
    # full-precision reference, so a single ulp of drift moves them by whole
    # units instead of hiding under the log's rounding.
    a = 100.0 + bar_index % 4
    b = 1.0 * bar_index

    k1 = ta.correlation(a, b, 4)
    # Constant second source: the variance product vanishes, and so does the
    # covariance, which is what makes the result 0.0 instead of na.
    k2 = ta.correlation(a, 100.0 + 0.0 * bar_index, 4)
    # Length 1: both variances and the covariance cancel exactly.
    k3 = ta.correlation(a, b, 1)

    # Perfectly correlated pairs on both sides of Pine's zero tolerance. The
    # covariance is 2 * s^2 * 1.25, so it lands below 1e-10 at s = 5e-6 -- where
    # the result is 0.0 even though the correlation is 1 -- and above it at
    # s = 1e-5, where the 1 comes through.
    lo = 0.000005 * (bar_index % 4)
    hi = 0.00001 * (bar_index % 4)
    k4 = ta.correlation(lo, lo * 2.0, 4)
    k5 = ta.correlation(hi, hi * 2.0, 4)

    # na gaps compact the rolling windows, which lets the result leave [-1, 1].
    g = na if bar_index % 3 == 0 else a
    k6 = ta.correlation(g, b, 4)

    r1 = -0.2 if bar_index == 8 else -0.6
    r6 = -36.51483716701107 if bar_index == 8 else -110.27480824437345

    if bar_index == 8 or bar_index == 9:
        log.info("{0}/{1}/{2}/{3}/{4}/{5}", (k1 - r1) * 1e17, k2, k3, k4, k5, (k6 - r6) * 1e14)


def __test_correlation_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Correlation TradingView-exact machine """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(12):
            next(run_iter)
