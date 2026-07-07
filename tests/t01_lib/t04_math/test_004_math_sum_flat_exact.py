"""
@pyne
"""
from pynecore import Persistent
from pynecore.lib import script, log, bar_index, math, ta


@script.indicator(title="Math Sum Flat Exact", shorttitle="math_sum_flat_exact")
def main():
    # Regression for the sliding-window drift class: the incremental Kahan
    # remove+add path used to carry residual rounding error from bars long
    # outside the window, so a window of equal values did not sum to exactly
    # n*v (a run of zeros summed to ~-3e-15) and ``sma(sma(x, 3), 3)`` of a
    # flat series flipped strict comparisons on last-bit noise (the TV
    # Technical Ratings ``kStochRsi < dStochRsi`` idiom fired spuriously).
    v: Persistent[float] = 0.0
    if bar_index < 10:
        v = float(bar_index) * 7.3 + 0.1  # noisy warmup charges the residue
    elif bar_index < 20:
        v = 100.0  # saturated flat run (stoch pegged at 100)
    else:
        v = 0.0  # flat run of zeros

    k = ta.sma(v, 3)
    d = ta.sma(k, 3)
    s = math.sum(v, 3)

    if bar_index >= 16:
        log.info("k_eq_d: {0}, sum_exact: {1}", k == d, s == v * 3)


def __test_math_sum_flat_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ math.sum() / ta.sma() - flat windows are exact """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for i in range(30):
            next(run_iter)
