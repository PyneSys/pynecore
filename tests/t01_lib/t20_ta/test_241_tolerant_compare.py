"""
@pyne
"""
from pynecore.lib import script, log, bar_index, ta


@script.indicator(title="Tolerant Compare", shorttitle="tolerant_compare")
def main():
    # Pine compares floats with an absolute 1e-10 tolerance, but only SOME
    # builtins inherit it -- the rest are bit-exact. ``dsub`` is a step far
    # below that tolerance, ``dsup`` one far above it, so each pair below
    # separates the two behaviours. Every expected value comes from running
    # these exact series on TradingView (probes m547/m548, 2026-08-02).
    base = 100.0
    dsub = 1e-12
    dsup = 1e-8

    r = base if bar_index % 2 == 0 else base + dsub
    rc = base if bar_index % 2 == 0 else base + dsup
    u = base + dsub if bar_index % 3 == 1 else base
    uc = base + dsup if bar_index % 3 == 1 else base

    # Tolerant: a sub-tolerance step is not a rise/fall
    rise_sub = ta.rising(r, 1)
    fall_sub = ta.falling(r, 1)
    # Supra control: the same builtins still see a real step
    rise_sup = ta.rising(rc, 1)
    fall_sup = ta.falling(rc, 1)
    # Tolerant: sub-tolerance neighbours all count as "at or below"
    prank_sub = ta.percentrank(u, 5)
    prank_sup = ta.percentrank(uc, 5)
    # Tolerant: the momentum buckets collapse, so the oscillator is na
    cmo_sub = ta.cmo(r, 4)
    # Tolerant plus TradingView's empty-bucket guard
    mfi_sub = ta.mfi(r, 4)
    # Bit-exact: these must NOT gain the tolerance
    # Scaled so the sub-tolerance step survives the log's float formatting
    hi_sub = (ta.highest(u, 3) - base) * 1e12
    hib_sub = ta.highestbars(u, 3)
    xover_sub = ta.crossover(u, base)

    if 12 <= bar_index <= 17:
        log.info("rise {0}/{1} fall {2}/{3} prank {4}/{5} cmo {6} mfi {7} "
                 "hi {8} hib {9} xover {10}",
                 rise_sub, rise_sup, fall_sub, fall_sup, prank_sub, prank_sup,
                 cmo_sub, mfi_sub, hi_sub, hib_sub, xover_sub)


def __test_tolerant_compare__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ Tolerant vs bit-exact builtins """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(20):
            next(run_iter)
