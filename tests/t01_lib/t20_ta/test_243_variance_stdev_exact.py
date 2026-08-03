"""
@pyne
"""
from pynecore.lib import script, log, bar_index, ta


@script.indicator(title="Variance Stdev Exact", shorttitle="variance_stdev_exact")
def main():
    # TradingView parity for the variance/stdev/dev machine. Every expected
    # value below is TradingView's own log output for these exact series
    # (probe m556, 2026-08-03). The scaled terms pin the last bits: they
    # amplify the ulp-level rounding pattern of TV's exact expression shapes
    # (q / L - m * m, the distributed unbiased form, sqrt, and the zero clamp),
    # so any deviation from the measured machine shifts them by whole units.
    c = 100.0 + bar_index % 4
    v = ta.variance(c, 4)
    vu = ta.variance(c, 4, False)
    s = ta.stdev(c, 4)
    su = ta.stdev(c, 4, False)
    d = ta.dev(c, 4)

    # Catastrophic cancellation: the raw expression lands on the ulp grid of
    # the squared scale (ulp(1e16) = 2), and the clamp keeps it non-negative.
    cc = 100000000.0 + (bar_index % 2) * 0.001
    vc = ta.variance(cc, 4)
    sc = ta.stdev(cc, 4)

    if bar_index == 8:
        log.info("{0}/{1}/{2}/{3}/{4}/{5}/{6}",
                 v,
                 (vu * 3.0 - 5.0) * 1e16,
                 (s * s - 1.25) * 1e16,
                 (su * su * 3.0 - 5.0) * 1e16,
                 d,
                 vc,
                 (sc * sc - vc) * 1e16)


def __test_variance_stdev_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Variance / stdev / dev TradingView-exact machine """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(12):
            next(run_iter)
