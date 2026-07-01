"""
@pyne
"""
from pynecore.lib import script, log, bar_index, array, na


@script.indicator(title="Array Functions NA", shorttitle="arr_func_na")
def main():
    if bar_index == 0:
        # Interspersed na, real values [1, 3, 5]; TradingView ignores na
        a: list = array.from_items(1.0, na, 3.0, na, 5.0)
        log.info("avg: {0}", array.avg(a))
        log.info("sum: {0}", array.sum(a))
        log.info("min: {0}", array.min(a))
        log.info("max: {0}", array.max(a))
        log.info("range: {0}", array.range(a))
        log.info("median: {0}", array.median(a))
        log.info("variance: {0,number,#.############}", array.variance(a))
        log.info("variance_unbiased: {0,number,#.############}", array.variance(a, False))
        log.info("stdev: {0,number,#.############}", array.stdev(a))
        log.info("stdev_unbiased: {0,number,#.############}", array.stdev(a, False))

        # mode over real values [4, 4, 7]
        m = array.from_items(4.0, na, 4.0, 7.0, na)
        log.info("mode: {0}", array.mode(m))

        # covariance: na dropped pairwise -> pairs (1,2) and (5,6)
        c1: list = array.from_items(1.0, na, 3.0, 5.0)
        c2: list = array.from_items(2.0, 4.0, na, 6.0)
        log.info("cov: {0,number,#.############}", array.covariance(c1, c2))
        log.info("cov_unbiased: {0,number,#.############}", array.covariance(c1, c2, False))

        # single real value survives; stdev/variance of one element are 0
        s: list = array.from_items(na, 6.0, na)
        log.info("s_avg: {0}", array.avg(s))
        log.info("s_stdev: {0}", array.stdev(s))
        log.info("s_variance: {0}", array.variance(s))

        # all-na array reduces to na
        z: list = array.from_items(na, na)
        log.info("z_avg: {0}", array.avg(z))
        log.info("z_sum: {0}", array.sum(z))
        log.info("z_min: {0}", array.min(z))


def __test_array_func_na__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Functions - na handling (ignore na like TradingView) """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)
