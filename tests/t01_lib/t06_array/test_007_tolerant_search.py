"""
@pyne
"""
from pynecore.lib import script, log, bar_index, array


@script.indicator(title="Array Tolerant Search", shorttitle="array_tolerant_search")
def main():
    # TradingView's array searches compare tolerantly, but ``binary_search``
    # and the percentile/statistics reductions are bit-exact -- the two
    # disagree on values closer than 1e-10, which is TradingView's own
    # behaviour. Expected values measured with these exact arrays (probes
    # m548/m551, 2026-08-02).
    base = 100.0
    dsub = 1e-12
    dsup = 1e-8

    dup = array.from_items(base, base + 1.0, base, base + 2.0)
    sorted_arr = array.from_items(base, base + 1.0, base + 2.0)
    pct = array.from_items(base, base + dsub, base + 5.0)

    if bar_index == 3:
        log.info("idx {0} lastidx {1} lastidx_sup {2} includes {3} bsearch {4} pnr {5}",
                 array.indexof(dup, base + dsub),
                 array.lastindexof(dup, base + dsub),
                 array.lastindexof(dup, base + dsup),
                 array.includes(dup, base + dsub),
                 array.binary_search(sorted_arr, base + dsub),
                 (array.percentile_nearest_rank(pct, 50) - base) * 1e12)


def __test_array_tolerant_search__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Tolerant array search vs bit-exact binary search """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(6):
            next(run_iter)
