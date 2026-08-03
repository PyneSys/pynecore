"""
@pyne
"""
from pynecore.lib import script, log, bar_index, ta


@script.indicator(title="Percentile Change Exact", shorttitle="percentile_change_exact")
def main():
    # TradingView parity for the percentile rank/interpolation formulas and for
    # ta.change's exactness. Every expected value below is TradingView's own
    # plotted output for these exact series (probes m552/m554, 2026-08-03).
    base = 100.0

    # A period-4 ramp: the window is always {base, base+1, base+2, base+3}, so
    # the percentage sweep pins the rank formula (nearest rank) and the
    # interpolation position (linear interpolation) without any tie in play.
    c = base + (bar_index % 4)

    # Linear interpolation walks the ranks in half steps: pos = n*p/100 + 0.5
    li0 = ta.percentile_linear_interpolation(c, 4, 0) - base
    li12 = ta.percentile_linear_interpolation(c, 4, 12.5) - base
    li25 = ta.percentile_linear_interpolation(c, 4, 25) - base
    li37 = ta.percentile_linear_interpolation(c, 4, 37.5) - base
    li50 = ta.percentile_linear_interpolation(c, 4, 50) - base
    li100 = ta.percentile_linear_interpolation(c, 4, 100) - base

    # Nearest rank steps at every ceil(p*n/100) boundary
    nr0 = ta.percentile_nearest_rank(c, 4, 0) - base
    nr25 = ta.percentile_nearest_rank(c, 4, 25) - base
    nr26 = ta.percentile_nearest_rank(c, 4, 26) - base
    nr50 = ta.percentile_nearest_rank(c, 4, 50) - base
    nr51 = ta.percentile_nearest_rank(c, 4, 51) - base
    nr100 = ta.percentile_nearest_rank(c, 4, 100) - base

    # ta.change subtracts raw doubles -- no quantization. The base is small
    # enough that a 1e-15 step is exactly representable, so a round(x, 14)
    # anywhere in the chain would collapse ``chg_tiny`` to zero.
    small = 0.001
    tiny = small if bar_index % 2 == 0 else small + 1e-15
    big = base if bar_index % 2 == 0 else base + 1e-12
    chg_tiny = ta.change(tiny) * 1e15
    chg_big = ta.change(big) * 1e12

    if bar_index == 8:
        log.info("li {0}/{1}/{2}/{3}/{4}/{5} nr {6}/{7}/{8}/{9}/{10}/{11} "
                 "chg {12}/{13}",
                 li0, li12, li25, li37, li50, li100,
                 nr0, nr25, nr26, nr50, nr51, nr100,
                 chg_tiny, chg_big)


def __test_percentile_change_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Percentile rank formulas and ta.change exactness """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(12):
            next(run_iter)
