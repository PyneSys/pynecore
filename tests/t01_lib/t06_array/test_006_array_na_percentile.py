"""
@pyne
"""
from pynecore.lib import script, log, bar_index, array, na


@script.indicator(title="Array NA Percentile", shorttitle="arr_na_pct")
def main():
    if bar_index == 0:
        # real [10, 20, 30, 40], na at idx 1 and 4, size 6
        a: list = array.from_items(10.0, na, 20.0, 30.0, na, 40.0)

        # nearest_rank: na kept and sorted to the end, full length drives the
        # rank; a rank landing in the na tail yields na (TradingView-verified)
        log.info("pnr0: {0}", array.percentile_nearest_rank(a, 0))
        log.info("pnr25: {0}", array.percentile_nearest_rank(a, 25))
        log.info("pnr50: {0}", array.percentile_nearest_rank(a, 50))
        log.info("pnr60: {0}", array.percentile_nearest_rank(a, 60))
        log.info("pnr75: {0}", array.percentile_nearest_rank(a, 75))
        log.info("pnr100: {0}", array.percentile_nearest_rank(a, 100))

        # percentrank: na ignored in the <= count but counted in the length;
        # a na element at the index yields na (TradingView-verified)
        log.info("prank0: {0}", array.percentrank(a, 0))
        log.info("prank1na: {0}", array.percentrank(a, 1))
        log.info("prank2: {0}", array.percentrank(a, 2))
        log.info("prank5: {0}", array.percentrank(a, 5))

        # linear_interpolation with na: pos = n * p / 100 + 0.5 over the full
        # length, na sorted to the end. TradingView yields a value only for the
        # low clamp (p0 -> pos 0.5) or an exact integer rank (p25 -> pos 2.0),
        # and na for every fractional position (p10 -> pos 1.1, p50 -> pos 3.5)
        # even when neighbours are numeric, or an exact rank in the na tail
        # (p75 -> pos 5.0, p100 -> pos 6.5). TradingView-verified.
        log.info("pli_na0: {0}", array.percentile_linear_interpolation(a, 0))
        log.info("pli_na10: {0}", array.percentile_linear_interpolation(a, 10))
        log.info("pli_na25: {0}", array.percentile_linear_interpolation(a, 25))
        log.info("pli_na50: {0}", array.percentile_linear_interpolation(a, 50))
        log.info("pli_na75: {0}", array.percentile_linear_interpolation(a, 75))
        log.info("pli_na100: {0}", array.percentile_linear_interpolation(a, 100))

        # linear_interpolation without na: same pos = n * p / 100 + 0.5 formula,
        # interpolating normally between straddling ranks. p15/p25/p75 land on
        # fractional positions under the old n*p/100 formula and were wrong;
        # these values are TradingView-verified for [0..10.1], n=10.
        b: list = array.from_items(0.0, 1.1, 2.2, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1)
        log.info("pli_c10: {0}", array.percentile_linear_interpolation(b, 10))
        log.info("pli_c15: {0}", array.percentile_linear_interpolation(b, 15))
        log.info("pli_c25: {0}", array.percentile_linear_interpolation(b, 25))
        log.info("pli_c50: {0}", array.percentile_linear_interpolation(b, 50))
        log.info("pli_c75: {0}", array.percentile_linear_interpolation(b, 75))
        log.info("pli_c90: {0}", array.percentile_linear_interpolation(b, 90))


def __test_array_na_percentile__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Percentile functions - na handling """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        next(run_iter)
