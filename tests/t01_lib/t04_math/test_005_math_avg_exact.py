"""
@pyne
"""
from pynecore.lib import script, log, bar_index, math


@script.indicator(title="Math Avg Exact", shorttitle="math_avg_exact")
def main():
    # TradingView runs two different sums inside math.avg, and every reference
    # below is the full-precision value it produced for these exact arguments
    # (probes m569-m576, 2026-08-04). Each case is one where the two candidate
    # sums disagree, so a wrong branch shows up as a non-zero difference; the
    # differences are amplified by 1e15 because the log rounds.
    #
    # Two arguments are added plainly. This pair is the one that made the
    # ichimoku donchian midpoint diverge: the compensated sum lands one ulp
    # above TradingView here.
    a2 = math.avg(0.99521, 1.00504) - 1.000125
    # From three arguments on the terms go through a Kahan compensated sum with
    # the pending correction flushed back before the division. The first case is
    # a Connors RSI bar, where a plain sum is one ulp low; the second is a window
    # whose terms span four orders of magnitude, where bare Kahan without the
    # flush is the one that misses.
    a3 = math.avg(60.48816185259965, 66.50424923195895, 78.0) - 68.33080369485288
    b3 = math.avg(22363.666666666668, 1.067091, 67091000.5) - 22371121.74458589
    a4 = math.avg(22364.333333333332, 1.067093, 67093000.5,
                  -9584.714285714286) - 16776445.296535155
    a5 = math.avg(11540.0, 1.03462, 34620000.5, -4945.714285714285,
                  18.351718397760905) - 6925322.834410538
    # Heavy cancellation: the two leading terms wipe each other out, so the
    # result is carried entirely by the compensation.
    a6 = math.avg(31676000000.5, -31675999999.5, 4525.142857142857,
                  3.156864602077217e-05, 0.00031676,
                  -4525.142857141856) - 0.16672472160780671

    if bar_index == 0:
        log.info("{0}|{1}|{2}|{3}|{4}|{5}", a2 * 1e15, a3 * 1e15, b3 * 1e15,
                 a4 * 1e15, a5 * 1e15, a6 * 1e15)


def __test_math_avg_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ math.avg() - TradingView-exact two-branch sum """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)
