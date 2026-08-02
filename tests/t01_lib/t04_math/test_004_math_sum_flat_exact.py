"""
@pyne
"""
from pynecore import Persistent
from pynecore.lib import script, log, bar_index, math, ta


@script.indicator(title="Math Sum Flat Exact", shorttitle="math_sum_flat_exact")
def main():
    # TV-parity regression on flat windows: TV's rolling sum carries residual
    # rounding dust INTO a flat run (the zero window here displays ~-2.8e-14,
    # not exact 0.0), yet the ``sma(sma(x, 3), 3)`` chain of a flat series
    # still converges to ``k == d`` a few bars in (the TV Technical Ratings
    # ``kStochRsi < dStochRsi`` idiom). The expected log is real TV output
    # for this exact sequence (m466 probe, 2026-08-02).
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
